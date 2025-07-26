"""Authentication and user management API routes."""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from typing import List, Dict, Any

from .auth_models import (
    UserCreate, UserLogin, UserUpdate, UserResponse, Token,
    PasswordReset, PasswordResetConfirm, UserRole,
    Workspace, WorkspaceCreate, WorkspaceUpdate, WorkspaceInvite,
    WorkspaceResponse
)
from .auth_service import AuthService
from .workspace_service import WorkspaceService
from .auth_dependencies import (
    get_current_user, get_current_active_user, require_role,
    get_auth_service, get_optional_user
)

# Create router
auth_router = APIRouter(prefix="/auth", tags=["authentication"])
workspace_router = APIRouter(prefix="/workspaces", tags=["workspaces"])

# Global services
workspace_service = WorkspaceService()


# Authentication endpoints
@auth_router.post("/register", response_model=UserResponse)
async def register_user(
    user_create: UserCreate,
    auth_service: AuthService = Depends(get_auth_service)
):
    """Register a new user."""
    try:
        user = auth_service.create_user(user_create)
        return UserResponse(**user.dict())
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@auth_router.post("/login", response_model=Token)
async def login_user(
    user_login: UserLogin,
    auth_service: AuthService = Depends(get_auth_service)
):
    """Login user and return JWT token."""
    try:
        token = auth_service.login_user(user_login)
        return token
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


@auth_router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user = Depends(get_current_active_user)
):
    """Get current user profile."""
    return UserResponse(**current_user.dict())


@auth_router.put("/me", response_model=UserResponse)
async def update_user_profile(
    user_update: UserUpdate,
    current_user = Depends(get_current_active_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Update current user profile."""
    try:
        updated_user = auth_service.update_user_profile(
            current_user.id, 
            user_update.dict(exclude_unset=True)
        )
        return UserResponse(**updated_user.dict())
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@auth_router.post("/change-password")
async def change_password(
    current_password: str,
    new_password: str,
    current_user = Depends(get_current_active_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Change user password."""
    try:
        auth_service.change_user_password(
            current_user.id, 
            current_password, 
            new_password
        )
        return {"message": "Password changed successfully"}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@auth_router.post("/logout")
async def logout_user(current_user = Depends(get_current_active_user)):
    """Logout user (client should discard token)."""
    return {"message": "Logged out successfully"}


# Admin endpoints
@auth_router.get("/users", response_model=List[UserResponse])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    current_user = Depends(require_role(UserRole.ADMIN)),
    auth_service: AuthService = Depends(get_auth_service)
):
    """List all users (admin only)."""
    return auth_service.list_users(skip=skip, limit=limit)


@auth_router.get("/stats")
async def get_user_stats(
    current_user = Depends(require_role(UserRole.ADMIN)),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Get user statistics (admin only)."""
    return auth_service.get_user_stats()


@auth_router.post("/users/{user_id}/deactivate")
async def deactivate_user(
    user_id: str,
    current_user = Depends(require_role(UserRole.ADMIN)),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Deactivate user account (admin only)."""
    success = auth_service.deactivate_user(user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return {"message": "User deactivated successfully"}


# Workspace endpoints
@workspace_router.post("/", response_model=WorkspaceResponse)
async def create_workspace(
    workspace_create: WorkspaceCreate,
    current_user = Depends(get_current_active_user)
):
    """Create a new workspace."""
    try:
        workspace = workspace_service.create_workspace(workspace_create, current_user.id)
        return WorkspaceResponse(
            id=workspace.id,
            name=workspace.name,
            description=workspace.description,
            owner_id=workspace.owner_id,
            created_at=workspace.created_at,
            updated_at=workspace.updated_at,
            is_public=workspace.is_public,
            member_count=len(workspace.members),
            settings=workspace.settings
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@workspace_router.get("/", response_model=List[WorkspaceResponse])
async def get_user_workspaces(
    current_user = Depends(get_current_active_user)
):
    """Get all workspaces for current user."""
    return workspace_service.get_user_workspaces(current_user.id)


@workspace_router.get("/{workspace_id}", response_model=WorkspaceResponse)
async def get_workspace(
    workspace_id: str,
    current_user = Depends(get_current_active_user)
):
    """Get workspace details."""
    workspace = workspace_service.get_workspace(workspace_id)
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found"
        )
    
    # Check access permissions
    if not (workspace_service._is_member(workspace_id, current_user.id) or workspace.is_public):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    return WorkspaceResponse(
        id=workspace.id,
        name=workspace.name,
        description=workspace.description,
        owner_id=workspace.owner_id,
        created_at=workspace.created_at,
        updated_at=workspace.updated_at,
        is_public=workspace.is_public,
        member_count=len(workspace.members),
        settings=workspace.settings
    )


@workspace_router.put("/{workspace_id}", response_model=WorkspaceResponse)
async def update_workspace(
    workspace_id: str,
    workspace_update: WorkspaceUpdate,
    current_user = Depends(get_current_active_user)
):
    """Update workspace (owner/admin only)."""
    try:
        workspace = workspace_service.update_workspace(
            workspace_id, 
            workspace_update, 
            current_user.id
        )
        return WorkspaceResponse(
            id=workspace.id,
            name=workspace.name,
            description=workspace.description,
            owner_id=workspace.owner_id,
            created_at=workspace.created_at,
            updated_at=workspace.updated_at,
            is_public=workspace.is_public,
            member_count=len(workspace.members),
            settings=workspace.settings
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@workspace_router.delete("/{workspace_id}")
async def delete_workspace(
    workspace_id: str,
    current_user = Depends(get_current_active_user)
):
    """Delete workspace (owner only)."""
    try:
        success = workspace_service.delete_workspace(workspace_id, current_user.id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workspace not found"
            )
        return {"message": "Workspace deleted successfully"}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )


@workspace_router.get("/{workspace_id}/members")
async def get_workspace_members(
    workspace_id: str,
    current_user = Depends(get_current_active_user)
):
    """Get workspace members."""
    try:
        members = workspace_service.get_workspace_members(workspace_id, current_user.id)
        return {"members": members}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )


@workspace_router.post("/{workspace_id}/members")
async def add_workspace_member(
    workspace_id: str,
    user_id: str,
    role: UserRole = UserRole.VIEWER,
    current_user = Depends(get_current_active_user)
):
    """Add member to workspace."""
    try:
        workspace_service.add_member(workspace_id, user_id, role, current_user.id)
        return {"message": "Member added successfully"}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@workspace_router.delete("/{workspace_id}/members/{user_id}")
async def remove_workspace_member(
    workspace_id: str,
    user_id: str,
    current_user = Depends(get_current_active_user)
):
    """Remove member from workspace."""
    try:
        workspace_service.remove_member(workspace_id, user_id, current_user.id)
        return {"message": "Member removed successfully"}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@workspace_router.put("/{workspace_id}/members/{user_id}/role")
async def update_member_role(
    workspace_id: str,
    user_id: str,
    new_role: UserRole,
    current_user = Depends(get_current_active_user)
):
    """Update member role in workspace."""
    try:
        workspace_service.update_member_role(workspace_id, user_id, new_role, current_user.id)
        return {"message": "Member role updated successfully"}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@workspace_router.get("/search/{query}")
async def search_workspaces(
    query: str,
    limit: int = 20,
    current_user = Depends(get_current_active_user)
):
    """Search workspaces."""
    results = workspace_service.search_workspaces(query, current_user.id, limit)
    return {"results": results, "total": len(results), "query": query}