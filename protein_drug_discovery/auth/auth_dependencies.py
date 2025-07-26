"""FastAPI dependencies for authentication and authorization."""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional

from .auth_service import AuthService
from .auth_models import User, UserRole, TokenData

# Global auth service instance
auth_service = AuthService()

# HTTP Bearer token scheme
security = HTTPBearer()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current authenticated user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        token = credentials.credentials
        token_data = auth_service.verify_token(token)
        
        if token_data is None or token_data.user_id is None:
            raise credentials_exception
        
        user = auth_service.get_user_by_id(token_data.user_id)
        if user is None:
            raise credentials_exception
        
        return user
    except Exception:
        raise credentials_exception


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user (must be active)."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


def require_role(required_role: UserRole):
    """Dependency factory for role-based access control."""
    def role_checker(current_user: User = Depends(get_current_active_user)) -> User:
        # Admin can access everything
        if current_user.role == UserRole.ADMIN:
            return current_user
        
        # Check if user has required role or higher
        role_hierarchy = {
            UserRole.GUEST: 0,
            UserRole.VIEWER: 1,
            UserRole.RESEARCHER: 2,
            UserRole.ADMIN: 3
        }
        
        user_level = role_hierarchy.get(current_user.role, 0)
        required_level = role_hierarchy.get(required_role, 0)
        
        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role.value}"
            )
        
        return current_user
    
    return role_checker


async def get_optional_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[User]:
    """Get current user if authenticated, otherwise return None."""
    if not credentials:
        return None
    
    try:
        token = credentials.credentials
        token_data = auth_service.verify_token(token)
        
        if token_data is None or token_data.user_id is None:
            return None
        
        user = auth_service.get_user_by_id(token_data.user_id)
        return user if user and user.is_active else None
    except Exception:
        return None


def get_auth_service() -> AuthService:
    """Get the global auth service instance."""
    return auth_service