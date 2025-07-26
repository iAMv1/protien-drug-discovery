"""Authentication data models and schemas."""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import uuid


class UserRole(str, Enum):
    """User roles for role-based access control."""
    ADMIN = "admin"
    RESEARCHER = "researcher"
    VIEWER = "viewer"
    GUEST = "guest"


class User(BaseModel):
    """User model for database storage."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: EmailStr
    username: str
    hashed_password: str
    full_name: Optional[str] = None
    role: UserRole = UserRole.RESEARCHER
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    
    # Profile information
    institution: Optional[str] = None
    department: Optional[str] = None
    research_interests: List[str] = Field(default_factory=list)
    
    # Workspace memberships
    workspace_ids: List[str] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class UserCreate(BaseModel):
    """Schema for user registration."""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = Field(None, max_length=100)
    institution: Optional[str] = Field(None, max_length=100)
    department: Optional[str] = Field(None, max_length=100)
    research_interests: List[str] = Field(default_factory=list)


class UserUpdate(BaseModel):
    """Schema for user profile updates."""
    full_name: Optional[str] = Field(None, max_length=100)
    institution: Optional[str] = Field(None, max_length=100)
    department: Optional[str] = Field(None, max_length=100)
    research_interests: List[str] = Field(default_factory=list)


class UserResponse(BaseModel):
    """Schema for user data in API responses."""
    id: str
    email: EmailStr
    username: str
    full_name: Optional[str]
    role: UserRole
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login: Optional[datetime]
    institution: Optional[str]
    department: Optional[str]
    research_interests: List[str]
    workspace_ids: List[str]


class UserLogin(BaseModel):
    """Schema for user login."""
    username: str
    password: str


class Token(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class TokenData(BaseModel):
    """Token payload data."""
    user_id: Optional[str] = None
    username: Optional[str] = None
    role: Optional[UserRole] = None


class PasswordReset(BaseModel):
    """Schema for password reset request."""
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Schema for password reset confirmation."""
    token: str
    new_password: str = Field(..., min_length=8)


class Workspace(BaseModel):
    """Workspace model for team collaboration."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    owner_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Members and permissions
    members: List[Dict[str, Any]] = Field(default_factory=list)  # {user_id, role, joined_at}
    is_public: bool = False
    
    # Workspace settings
    settings: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class WorkspaceCreate(BaseModel):
    """Schema for workspace creation."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    is_public: bool = False


class WorkspaceUpdate(BaseModel):
    """Schema for workspace updates."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    is_public: Optional[bool] = None
    settings: Optional[Dict[str, Any]] = None


class WorkspaceInvite(BaseModel):
    """Schema for workspace invitations."""
    email: EmailStr
    role: UserRole = UserRole.VIEWER


class WorkspaceMember(BaseModel):
    """Schema for workspace member information."""
    user_id: str
    username: str
    email: EmailStr
    full_name: Optional[str]
    role: UserRole
    joined_at: datetime


class WorkspaceResponse(BaseModel):
    """Schema for workspace data in API responses."""
    id: str
    name: str
    description: Optional[str]
    owner_id: str
    created_at: datetime
    updated_at: datetime
    is_public: bool
    member_count: int
    settings: Dict[str, Any]