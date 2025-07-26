"""Authentication and authorization module for protein-drug discovery platform."""

from .auth_models import User, UserCreate, UserResponse, Token, TokenData, UserRole, Workspace, WorkspaceCreate
from .auth_service import AuthService
from .auth_dependencies import get_current_user, get_current_active_user, require_role
from .workspace_service import WorkspaceService

__all__ = [
    "User",
    "UserCreate", 
    "UserResponse",
    "Token",
    "TokenData",
    "UserRole",
    "Workspace",
    "WorkspaceCreate",
    "AuthService",
    "get_current_user",
    "get_current_active_user",
    "require_role",
    "WorkspaceService"
]