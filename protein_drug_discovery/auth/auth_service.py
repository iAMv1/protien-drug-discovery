"""Authentication service for user management and JWT token handling."""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import json
import os
from pathlib import Path

from jose import JWTError, jwt
from passlib.context import CryptContext

from .auth_models import User, UserCreate, UserLogin, Token, TokenData, UserRole, UserResponse


class AuthService:
    """Service for handling authentication operations."""
    
    def __init__(self, secret_key: str = None, algorithm: str = "HS256", access_token_expire_minutes: int = 30):
        self.SECRET_KEY = secret_key or os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
        self.ALGORITHM = algorithm
        self.ACCESS_TOKEN_EXPIRE_MINUTES = access_token_expire_minutes
        
        # Password hashing
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # In-memory storage for demo (replace with database in production)
        self.users_db: Dict[str, User] = {}
        self.users_by_email: Dict[str, str] = {}  # email -> user_id mapping
        self.users_by_username: Dict[str, str] = {}  # username -> user_id mapping
        
        # Load users from file if exists
        self._load_users()
        
        # Create default admin user if no users exist
        if not self.users_db:
            self._create_default_admin()
    
    def _load_users(self):
        """Load users from JSON file."""
        users_file = Path("data/users.json")
        if users_file.exists():
            try:
                with open(users_file, 'r') as f:
                    data = json.load(f)
                    for user_data in data.get('users', []):
                        user = User(**user_data)
                        self.users_db[user.id] = user
                        self.users_by_email[user.email] = user.id
                        self.users_by_username[user.username] = user.id
            except Exception as e:
                print(f"Error loading users: {e}")
    
    def _save_users(self):
        """Save users to JSON file."""
        users_file = Path("data/users.json")
        users_file.parent.mkdir(exist_ok=True)
        
        try:
            data = {
                'users': [user.dict() for user in self.users_db.values()]
            }
            with open(users_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving users: {e}")
    
    def _create_default_admin(self):
        """Create default admin user."""
        admin_user = UserCreate(
            email="admin@example.com",
            username="admin",
            password="admin123",
            full_name="System Administrator"
        )
        self.create_user(admin_user, role=UserRole.ADMIN)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash a password."""
        return self.pwd_context.hash(password)
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        user_id = self.users_by_username.get(username)
        if user_id:
            return self.users_db.get(user_id)
        return None
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        user_id = self.users_by_email.get(email)
        if user_id:
            return self.users_db.get(user_id)
        return None
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.users_db.get(user_id)
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username/password."""
        user = self.get_user_by_username(username)
        if not user:
            return None
        if not self.verify_password(password, user.hashed_password):
            return None
        
        # Update last login
        user.last_login = datetime.utcnow()
        self._save_users()
        
        return user
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.SECRET_KEY, algorithm=self.ALGORITHM)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.SECRET_KEY, algorithms=[self.ALGORITHM])
            user_id: str = payload.get("sub")
            username: str = payload.get("username")
            role: str = payload.get("role")
            
            if user_id is None:
                return None
            
            token_data = TokenData(
                user_id=user_id,
                username=username,
                role=UserRole(role) if role else None
            )
            return token_data
        except JWTError:
            return None
    
    def create_user(self, user_create: UserCreate, role: UserRole = UserRole.RESEARCHER) -> User:
        """Create a new user."""
        # Check if user already exists
        if self.get_user_by_email(user_create.email):
            raise ValueError("Email already registered")
        if self.get_user_by_username(user_create.username):
            raise ValueError("Username already taken")
        
        # Create user
        user = User(
            email=user_create.email,
            username=user_create.username,
            full_name=user_create.full_name,
            role=role,
            institution=user_create.institution,
            department=user_create.department,
            research_interests=user_create.research_interests,
            hashed_password=self.get_password_hash(user_create.password)
        )
        
        # Store user
        self.users_db[user.id] = user
        self.users_by_email[user.email] = user.id
        self.users_by_username[user.username] = user.id
        
        # Save to file
        self._save_users()
        
        return user
    
    def login_user(self, user_login: UserLogin) -> Token:
        """Login user and return JWT token."""
        user = self.authenticate_user(user_login.username, user_login.password)
        if not user:
            raise ValueError("Incorrect username or password")
        
        if not user.is_active:
            raise ValueError("User account is disabled")
        
        # Create access token
        access_token_expires = timedelta(minutes=self.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = self.create_access_token(
            data={
                "sub": user.id,
                "username": user.username,
                "role": user.role.value
            },
            expires_delta=access_token_expires
        )
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=self.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=UserResponse(**user.dict())
        )
    
    def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> User:
        """Update user profile."""
        user = self.get_user_by_id(user_id)
        if not user:
            raise ValueError("User not found")
        
        # Update allowed fields
        allowed_fields = ['full_name', 'institution', 'department', 'research_interests']
        for field, value in updates.items():
            if field in allowed_fields and hasattr(user, field):
                setattr(user, field, value)
        
        user.updated_at = datetime.utcnow()
        self._save_users()
        
        return user
    
    def change_user_password(self, user_id: str, current_password: str, new_password: str) -> bool:
        """Change user password."""
        user = self.get_user_by_id(user_id)
        if not user:
            raise ValueError("User not found")
        
        if not self.verify_password(current_password, user.hashed_password):
            raise ValueError("Current password is incorrect")
        
        user.hashed_password = self.get_password_hash(new_password)
        user.updated_at = datetime.utcnow()
        self._save_users()
        
        return True
    
    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate user account."""
        user = self.get_user_by_id(user_id)
        if not user:
            return False
        
        user.is_active = False
        user.updated_at = datetime.utcnow()
        self._save_users()
        
        return True
    
    def list_users(self, skip: int = 0, limit: int = 100) -> List[UserResponse]:
        """List all users (admin only)."""
        users = list(self.users_db.values())[skip:skip + limit]
        return [UserResponse(**user.dict()) for user in users]
    
    def get_user_stats(self) -> Dict[str, Any]:
        """Get user statistics."""
        total_users = len(self.users_db)
        active_users = sum(1 for user in self.users_db.values() if user.is_active)
        role_counts = {}
        
        for user in self.users_db.values():
            role = user.role.value
            role_counts[role] = role_counts.get(role, 0) + 1
        
        return {
            "total_users": total_users,
            "active_users": active_users,
            "inactive_users": total_users - active_users,
            "role_distribution": role_counts
        }