"""Workspace service for team collaboration and data sharing."""

from datetime import datetime
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

from .auth_models import (
    Workspace, WorkspaceCreate, WorkspaceUpdate, WorkspaceInvite,
    WorkspaceMember, WorkspaceResponse, User, UserRole
)


class WorkspaceService:
    """Service for managing team workspaces and collaboration."""
    
    def __init__(self):
        # In-memory storage for demo (replace with database in production)
        self.workspaces_db: Dict[str, Workspace] = {}
        self.workspace_members: Dict[str, List[Dict[str, Any]]] = {}  # workspace_id -> members
        
        # Load workspaces from file if exists
        self._load_workspaces()
    
    def _load_workspaces(self):
        """Load workspaces from JSON file."""
        workspaces_file = Path("data/workspaces.json")
        if workspaces_file.exists():
            try:
                with open(workspaces_file, 'r') as f:
                    data = json.load(f)
                    for workspace_data in data.get('workspaces', []):
                        workspace = Workspace(**workspace_data)
                        self.workspaces_db[workspace.id] = workspace
                        self.workspace_members[workspace.id] = workspace.members
            except Exception as e:
                print(f"Error loading workspaces: {e}")
    
    def _save_workspaces(self):
        """Save workspaces to JSON file."""
        workspaces_file = Path("data/workspaces.json")
        workspaces_file.parent.mkdir(exist_ok=True)
        
        try:
            data = {
                'workspaces': [workspace.dict() for workspace in self.workspaces_db.values()]
            }
            with open(workspaces_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving workspaces: {e}")
    
    def create_workspace(self, workspace_create: WorkspaceCreate, owner_id: str) -> Workspace:
        """Create a new workspace."""
        workspace = Workspace(
            name=workspace_create.name,
            description=workspace_create.description,
            owner_id=owner_id,
            is_public=workspace_create.is_public
        )
        
        # Add owner as admin member
        workspace.members = [{
            "user_id": owner_id,
            "role": UserRole.ADMIN.value,
            "joined_at": datetime.utcnow().isoformat()
        }]
        
        # Store workspace
        self.workspaces_db[workspace.id] = workspace
        self.workspace_members[workspace.id] = workspace.members
        
        # Save to file
        self._save_workspaces()
        
        return workspace
    
    def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
        """Get workspace by ID."""
        return self.workspaces_db.get(workspace_id)
    
    def update_workspace(self, workspace_id: str, updates: WorkspaceUpdate, user_id: str) -> Workspace:
        """Update workspace (owner/admin only)."""
        workspace = self.get_workspace(workspace_id)
        if not workspace:
            raise ValueError("Workspace not found")
        
        # Check permissions
        if not self._can_manage_workspace(workspace_id, user_id):
            raise ValueError("Insufficient permissions to update workspace")
        
        # Update fields
        if updates.name is not None:
            workspace.name = updates.name
        if updates.description is not None:
            workspace.description = updates.description
        if updates.is_public is not None:
            workspace.is_public = updates.is_public
        if updates.settings is not None:
            workspace.settings.update(updates.settings)
        
        workspace.updated_at = datetime.utcnow()
        self._save_workspaces()
        
        return workspace
    
    def delete_workspace(self, workspace_id: str, user_id: str) -> bool:
        """Delete workspace (owner only)."""
        workspace = self.get_workspace(workspace_id)
        if not workspace:
            return False
        
        # Only owner can delete
        if workspace.owner_id != user_id:
            raise ValueError("Only workspace owner can delete workspace")
        
        # Remove workspace
        del self.workspaces_db[workspace_id]
        if workspace_id in self.workspace_members:
            del self.workspace_members[workspace_id]
        
        self._save_workspaces()
        return True
    
    def add_member(self, workspace_id: str, user_id: str, role: UserRole = UserRole.VIEWER, inviter_id: str = None) -> bool:
        """Add member to workspace."""
        workspace = self.get_workspace(workspace_id)
        if not workspace:
            raise ValueError("Workspace not found")
        
        # Check if user is already a member
        if self._is_member(workspace_id, user_id):
            raise ValueError("User is already a member of this workspace")
        
        # Check permissions (admin/owner can add members)
        if inviter_id and not self._can_manage_workspace(workspace_id, inviter_id):
            raise ValueError("Insufficient permissions to add members")
        
        # Add member
        member_data = {
            "user_id": user_id,
            "role": role.value,
            "joined_at": datetime.utcnow().isoformat()
        }
        
        workspace.members.append(member_data)
        self.workspace_members[workspace_id] = workspace.members
        workspace.updated_at = datetime.utcnow()
        
        self._save_workspaces()
        return True
    
    def remove_member(self, workspace_id: str, user_id: str, remover_id: str) -> bool:
        """Remove member from workspace."""
        workspace = self.get_workspace(workspace_id)
        if not workspace:
            raise ValueError("Workspace not found")
        
        # Can't remove owner
        if user_id == workspace.owner_id:
            raise ValueError("Cannot remove workspace owner")
        
        # Check permissions
        if not self._can_manage_workspace(workspace_id, remover_id) and remover_id != user_id:
            raise ValueError("Insufficient permissions to remove member")
        
        # Remove member
        workspace.members = [m for m in workspace.members if m["user_id"] != user_id]
        self.workspace_members[workspace_id] = workspace.members
        workspace.updated_at = datetime.utcnow()
        
        self._save_workspaces()
        return True
    
    def update_member_role(self, workspace_id: str, user_id: str, new_role: UserRole, updater_id: str) -> bool:
        """Update member role in workspace."""
        workspace = self.get_workspace(workspace_id)
        if not workspace:
            raise ValueError("Workspace not found")
        
        # Can't change owner role
        if user_id == workspace.owner_id:
            raise ValueError("Cannot change workspace owner role")
        
        # Check permissions
        if not self._can_manage_workspace(workspace_id, updater_id):
            raise ValueError("Insufficient permissions to update member role")
        
        # Update role
        for member in workspace.members:
            if member["user_id"] == user_id:
                member["role"] = new_role.value
                break
        else:
            raise ValueError("User is not a member of this workspace")
        
        workspace.updated_at = datetime.utcnow()
        self._save_workspaces()
        return True
    
    def get_user_workspaces(self, user_id: str) -> List[WorkspaceResponse]:
        """Get all workspaces for a user."""
        user_workspaces = []
        
        for workspace in self.workspaces_db.values():
            # Check if user is a member or if workspace is public
            if self._is_member(workspace.id, user_id) or workspace.is_public:
                user_workspaces.append(WorkspaceResponse(
                    id=workspace.id,
                    name=workspace.name,
                    description=workspace.description,
                    owner_id=workspace.owner_id,
                    created_at=workspace.created_at,
                    updated_at=workspace.updated_at,
                    is_public=workspace.is_public,
                    member_count=len(workspace.members),
                    settings=workspace.settings
                ))
        
        return user_workspaces
    
    def get_workspace_members(self, workspace_id: str, user_id: str) -> List[Dict[str, Any]]:
        """Get workspace members (members only)."""
        workspace = self.get_workspace(workspace_id)
        if not workspace:
            raise ValueError("Workspace not found")
        
        # Check if user can view members
        if not self._is_member(workspace_id, user_id) and not workspace.is_public:
            raise ValueError("Access denied")
        
        return workspace.members
    
    def _is_member(self, workspace_id: str, user_id: str) -> bool:
        """Check if user is a member of workspace."""
        workspace = self.get_workspace(workspace_id)
        if not workspace:
            return False
        
        return any(member["user_id"] == user_id for member in workspace.members)
    
    def _can_manage_workspace(self, workspace_id: str, user_id: str) -> bool:
        """Check if user can manage workspace (owner or admin)."""
        workspace = self.get_workspace(workspace_id)
        if not workspace:
            return False
        
        # Owner can always manage
        if workspace.owner_id == user_id:
            return True
        
        # Check if user is admin member
        for member in workspace.members:
            if member["user_id"] == user_id and member["role"] == UserRole.ADMIN.value:
                return True
        
        return False
    
    def get_member_role(self, workspace_id: str, user_id: str) -> Optional[UserRole]:
        """Get user's role in workspace."""
        workspace = self.get_workspace(workspace_id)
        if not workspace:
            return None
        
        for member in workspace.members:
            if member["user_id"] == user_id:
                return UserRole(member["role"])
        
        return None
    
    def search_workspaces(self, query: str, user_id: str, limit: int = 20) -> List[WorkspaceResponse]:
        """Search public workspaces or user's workspaces."""
        results = []
        query_lower = query.lower()
        
        for workspace in self.workspaces_db.values():
            # Check if user can see this workspace
            if not (self._is_member(workspace.id, user_id) or workspace.is_public):
                continue
            
            # Check if workspace matches query
            if (query_lower in workspace.name.lower() or 
                (workspace.description and query_lower in workspace.description.lower())):
                results.append(WorkspaceResponse(
                    id=workspace.id,
                    name=workspace.name,
                    description=workspace.description,
                    owner_id=workspace.owner_id,
                    created_at=workspace.created_at,
                    updated_at=workspace.updated_at,
                    is_public=workspace.is_public,
                    member_count=len(workspace.members),
                    settings=workspace.settings
                ))
            
            if len(results) >= limit:
                break
        
        return results