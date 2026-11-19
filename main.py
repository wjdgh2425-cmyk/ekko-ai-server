
import os
import secrets
import hashlib
import hmac

from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from jose import jwt, JWTError

from sqlalchemy import (
    create_engine,
    String,
    Boolean,
    DateTime,
    Integer,
    select,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    sessionmaker,
    Session,
)

from openai import OpenAI


# =====================================
# 1. 환경 변수
# =====================================

DATABASE_URL = os.environ.get("DATABASE_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
JWT_SECRET = os.environ.get("JWT_SECRET")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL 환경 변수가 설정되지 않았습니다.")

if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET 환경 변수가 설정되지 않았습니다.")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

# Railway PostgreSQL 주소 호환
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgres://",
        "postgresql+psycopg://",
        1,
    )
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgresql://",
        "postgresql+psycopg://",
        1,
    )


# =====================================
# 2. 데이터베이스 연결
# =====================================

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    username: Mapped[str] = mapped_column(
        String(30),
        unique=True,
        index=True,
        nullable=False,
    )

    password_hash: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    password_salt: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    is_premium: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    payment_status: Mapped[str] = mapped_column(
        String(20),
        default="none",
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =====================================
# 3. 비밀번호 보안
# =====================================

def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        310000,
    ).hex()


def create_password(password: str):
    salt = secrets.token_hex(16)
    password_hash = hash_password(password, salt)
    return password_hash, salt


def verify_password(password: str, user: User) -> bool:
    check_hash = hash_password(password, user.password_salt)
    return hmac.compare_digest(
        check_hash,
        user.password_hash,
    )


# =====================================
# 4. JWT 로그인 인증
# =====================================

security = HTTPBearer()

TOKEN_EXPIRE_MINUTES = 60 * 24


def create_access_token(user_id: int):
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=TOKEN_EXPIRE_MINUTES
    )

    payload = {
        "sub": str(user_id),
        "exp": expire,
    }

    return jwt.encode(
        payload,
        JWT_SECRET,
        algorithm="HS256",
    )


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
):
    token = credentials.credentials

    try:
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=["HS256"],
        )

        user_id = int(payload.get("sub"))

    except (JWTError, TypeError, ValueError):
        raise HTTPException(
            status_code=401,
            detail="로그인이 만료되었거나 유효하지 않습니다.",
        )

    user = db.get(User, user_id)

    if not user:
        raise HTTPException(
            status_code=401,
            detail="사용자를 찾을 수 없습니다.",
        )

    return user


# =====================================
# 5. 요청 데이터
# =====================================

class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=30)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)


# =====================================
# 6. FastAPI 서버
# =====================================

app = FastAPI(
    title="EKKO AI Server",
    description="EKKO AI 공식 백엔드 서버",
    version="1.0.0",
)


@app.get("/")
def home():
    return {
        "message": "EKKO AI 서버가 정상 작동 중입니다.",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "EKKO AI",
    }


# =====================================
# 7. 회원가입
# =====================================

@app.post("/auth/register")
def register(
    request: RegisterRequest,
    db: Session = Depends(get_db),
):
    username = request.username.strip()

    if not username:
        raise HTTPException(
            status_code=400,
            detail="아이디를 입력해 주세요.",
        )

    existing_user = db.scalar(
        select(User).where(User.username == username)
    )

    if existing_user:
        raise HTTPException(
            status_code=409,
            detail="이미 사용 중인 아이디입니다.",
        )

    password_hash, salt = create_password(request.password)

    user = User(
        username=username,
        password_hash=password_hash,
        password_salt=salt,
        is_premium=False,
        payment_status="none",
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return {
        "success": True,
        "message": "회원가입이 완료되었습니다.",
        "username": user.username,
    }


# =====================================
# 8. 로그인
# =====================================

@app.post("/auth/login")
def login(
    request: LoginRequest,
    db: Session = Depends(get_db),
):
    user = db.scalar(
        select(User).where(
            User.username == request.username.strip()
        )
    )

    if not user or not verify_password(request.password, user):
        raise HTTPException(
            status_code=401,
            detail="아이디 또는 비밀번호가 올바르지 않습니다.",
        )

    access_token = create_access_token(user.id)

    return {
        "success": True,
        "access_token": access_token,
        "token_type": "bearer",
        "username": user.username,
        "is_premium": user.is_premium,
    }


# =====================================
# 9. 내 계정 정보
# =====================================

@app.get("/me")
def my_account(
    user: User = Depends(get_current_user),
):
    return {
        "id": user.id,
        "username": user.username,
        "is_premium": user.is_premium,
        "payment_status": user.payment_status,
        "created_at": user.created_at.isoformat(),
    }


# =====================================
# 10. EKKO AI 채팅
# =====================================

@app.post("/chat")
def chat(
    request: ChatRequest,
    user: User = Depends(get_current_user),
):
    # 프리미엄 사용자만 AI 채팅 가능
    if not user.is_premium:
        raise HTTPException(
            status_code=403,
            detail="EKKO AI PREMIUM 구독이 필요합니다.",
        )

    try:
        client = OpenAI(
            api_key=OPENAI_API_KEY,
        )

        response = client.responses.create(
            model="gpt-4o-mini",
            instructions=(
                "너는 EKKO AI라는 이름의 친절한 AI 비서야. "
                "사용자의 질문에 정확하고 자연스러운 한국어로 답변해 줘."
            ),
            input=request.message,
            max_output_tokens=1200,
        )

        return {
            "success": True,
            "answer": response.output_text,
        }

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="AI 서버와 통신하는 중 오류가 발생했습니다.",
        )
