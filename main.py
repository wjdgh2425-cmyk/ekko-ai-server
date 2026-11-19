
import os
import secrets
import hashlib
import hmac

from datetime import datetime, timedelta, timezone

from fastapi import (
    FastAPI, Depends, HTTPException, Header
)
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from jose import jwt, JWTError

from sqlalchemy import (
    create_engine, String, Boolean, DateTime,
    Integer, ForeignKey, select, inspect, text
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column,
    sessionmaker, Session
)

from openai import OpenAI


# =====================================
# 1. 환경 변수
# =====================================

DATABASE_URL = os.environ.get("DATABASE_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
JWT_SECRET = os.environ.get("JWT_SECRET")
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY")

BANK_NAME = os.environ.get("EKKO_BANK_NAME", "")
BANK_ACCOUNT = os.environ.get("EKKO_BANK_ACCOUNT", "")
BANK_HOLDER = os.environ.get("EKKO_BANK_HOLDER", "")

SUBSCRIPTION_PRICE = 9900
SUBSCRIPTION_DAYS = 30

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL 환경 변수가 설정되지 않았습니다.")

if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET 환경 변수가 설정되지 않았습니다.")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgres://", "postgresql+psycopg://", 1
    )
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgresql://", "postgresql+psycopg://", 1
    )


# =====================================
# 2. 데이터베이스
# =====================================

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    username: Mapped[str] = mapped_column(
        String(30), unique=True, index=True, nullable=False
    )

    password_hash: Mapped[str] = mapped_column(
        String(128), nullable=False
    )

    password_salt: Mapped[str] = mapped_column(
        String(64), nullable=False
    )

    is_premium: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )

    payment_status: Mapped[str] = mapped_column(
        String(20), default="none", nullable=False
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class PaymentRequest(Base):
    __tablename__ = "payment_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"), nullable=False, index=True
    )

    depositor_name: Mapped[str] = mapped_column(
        String(40), nullable=False
    )

    amount: Mapped[int] = mapped_column(
        Integer, default=SUBSCRIPTION_PRICE, nullable=False
    )

    status: Mapped[str] = mapped_column(
        String(20), default="pending", nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class Subscription(Base):
    __tablename__ = "subscriptions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"), nullable=False, index=True
    )

    payment_request_id: Mapped[int] = mapped_column(
        ForeignKey("payment_requests.id"), nullable=False, unique=True
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


Base.metadata.create_all(bind=engine)


# 기존 회원 테이블에 계정 활성화 컬럼이 없으면 추가
# 기존 회원 정보와 데이터는 삭제하지 않습니다.
existing_columns = {
    column["name"]
    for column in inspect(engine).get_columns("users")
}

if "is_active" not in existing_columns:
    with engine.begin() as connection:
        connection.execute(
            text(
                "ALTER TABLE users "
                "ADD COLUMN is_active BOOLEAN NOT NULL DEFAULT TRUE"
            )
        )


def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


# =====================================
# 3. 구독 상태
# =====================================

def get_subscription_state(db: Session, user: User):

    now = datetime.now(timezone.utc)

    active = db.scalar(
        select(Subscription)
        .where(
            Subscription.user_id == user.id,
            Subscription.expires_at > now,
        )
        .order_by(Subscription.expires_at.desc())
    )

    latest = db.scalar(
        select(Subscription)
        .where(Subscription.user_id == user.id)
        .order_by(Subscription.expires_at.desc())
    )

    pending = db.scalar(
        select(PaymentRequest)
        .where(
            PaymentRequest.user_id == user.id,
            PaymentRequest.status == "pending",
        )
        .order_by(PaymentRequest.created_at.desc())
    )

    if active:
        status = "active"
    elif pending:
        status = "pending"
    elif latest:
        status = "expired"
    else:
        status = "none"

    return {
        "is_premium": active is not None,
        "payment_status": status,
        "subscription": active,
        "pending_request": pending,
    }


def require_active_subscription(user: User, db: Session):

    if not user.is_active:
        raise HTTPException(
            status_code=403,
            detail="정지된 계정입니다. 관리자에게 문의해 주세요.",
        )

    state = get_subscription_state(db, user)

    if not state["is_premium"]:
        raise HTTPException(
            status_code=403,
            detail="EKKO AI PREMIUM 구독이 필요합니다.",
        )

    return state["subscription"]


def make_aware(value: datetime):
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


# =====================================
# 4. 비밀번호
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
    return hash_password(password, salt), salt


def verify_password(password: str, user: User) -> bool:
    check_hash = hash_password(password, user.password_salt)
    return hmac.compare_digest(check_hash, user.password_hash)


# =====================================
# 5. JWT 인증
# =====================================

security = HTTPBearer()
TOKEN_EXPIRE_MINUTES = 60 * 24


def create_access_token(user_id: int):

    expire = datetime.now(timezone.utc) + timedelta(
        minutes=TOKEN_EXPIRE_MINUTES
    )

    return jwt.encode(
        {"sub": str(user_id), "exp": expire},
        JWT_SECRET,
        algorithm="HS256",
    )


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
):

    try:
        payload = jwt.decode(
            credentials.credentials,
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

    if not user.is_active:
        raise HTTPException(
            status_code=403,
            detail="정지된 계정입니다. 관리자에게 문의해 주세요.",
        )

    return user


# =====================================
# 6. 관리자 인증
# =====================================

def require_admin(
    admin_key: str | None = Header(
        default=None,
        alias="X-Admin-Key",
    ),
):

    if not ADMIN_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="관리자 인증 키가 설정되지 않았습니다.",
        )

    if not admin_key or not hmac.compare_digest(
        admin_key, ADMIN_API_KEY
    ):
        raise HTTPException(
            status_code=403,
            detail="관리자 인증에 실패했습니다.",
        )

    return True


# =====================================
# 7. 요청 모델
# =====================================

class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=30)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)


class TransferRequestCreate(BaseModel):
    depositor_name: str = Field(min_length=2, max_length=40)


class AdminGrantRequest(BaseModel):
    days: int = Field(default=30, ge=1, le=3650)


# =====================================
# 8. FastAPI
# =====================================

app = FastAPI(
    title="EKKO AI Server",
    description="EKKO AI 공식 백엔드 서버",
    version="4.0.0",
)


@app.get("/")
def home():
    return {
        "message": "EKKO AI 서버가 정상 작동 중입니다.",
        "version": "4.0.0",
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "EKKO AI"}


# =====================================
# 9. 회원가입
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

    existing = db.scalar(
        select(User).where(User.username == username)
    )

    if existing:
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
        is_active=True,
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
# 10. 로그인
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

    if not user.is_active:
        raise HTTPException(
            status_code=403,
            detail="정지된 계정입니다. 관리자에게 문의해 주세요.",
        )

    state = get_subscription_state(db, user)

    return {
        "success": True,
        "access_token": create_access_token(user.id),
        "token_type": "bearer",
        "username": user.username,
        "is_premium": state["is_premium"],
    }


# =====================================
# 11. 내 계정
# =====================================

@app.get("/me")
def my_account(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):

    state = get_subscription_state(db, user)
    subscription = state["subscription"]

    return {
        "id": user.id,
        "username": user.username,
        "is_premium": state["is_premium"],
        "payment_status": state["payment_status"],
        "subscription_expires_at": (
            subscription.expires_at.isoformat()
            if subscription else None
        ),
        "created_at": user.created_at.isoformat(),
    }


# =====================================
# 12. 구독 정보
# =====================================

@app.get("/subscription/info")
def subscription_info(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):

    state = get_subscription_state(db, user)
    pending = state["pending_request"]
    subscription = state["subscription"]

    return {
        "success": True,
        "price": SUBSCRIPTION_PRICE,
        "duration_days": SUBSCRIPTION_DAYS,
        "bank_name": BANK_NAME,
        "bank_account": BANK_ACCOUNT,
        "bank_holder": BANK_HOLDER,
        "is_premium": state["is_premium"],
        "payment_status": state["payment_status"],
        "pending_request_id": pending.id if pending else None,
        "subscription_expires_at": (
            subscription.expires_at.isoformat()
            if subscription else None
        ),
    }


# =====================================
# 13. 계좌이체 신청
# =====================================

@app.post("/subscription/transfer-request")
def create_transfer_request(
    request: TransferRequestCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):

    if not BANK_NAME or not BANK_ACCOUNT or not BANK_HOLDER:
        raise HTTPException(
            status_code=503,
            detail="계좌 정보가 아직 설정되지 않았습니다.",
        )

    state = get_subscription_state(db, user)

    if state["is_premium"]:
        raise HTTPException(
            status_code=409,
            detail="이미 프리미엄 구독을 이용 중입니다.",
        )

    if state["pending_request"]:
        raise HTTPException(
            status_code=409,
            detail="이미 입금 확인 대기 중인 신청이 있습니다.",
        )

    depositor_name = request.depositor_name.strip()

    if not depositor_name:
        raise HTTPException(
            status_code=400,
            detail="입금자명을 입력해 주세요.",
        )

    payment_request = PaymentRequest(
        user_id=user.id,
        depositor_name=depositor_name,
        amount=SUBSCRIPTION_PRICE,
        status="pending",
    )

    db.add(payment_request)
    db.commit()
    db.refresh(payment_request)

    return {
        "success": True,
        "message": "구독 신청이 완료되었습니다.",
        "request_id": payment_request.id,
        "status": payment_request.status,
        "amount": payment_request.amount,
    }


# =====================================
# 14. 관리자 API - 신청 목록
# =====================================

@app.get("/admin/subscription-requests")
def admin_subscription_requests(
    status: str = "pending",
    admin: bool = Depends(require_admin),
    db: Session = Depends(get_db),
):

    if status not in ["pending", "approved", "rejected", "all"]:
        raise HTTPException(
            status_code=400,
            detail="올바르지 않은 상태입니다.",
        )

    query = (
        select(PaymentRequest, User)
        .join(User, PaymentRequest.user_id == User.id)
        .order_by(PaymentRequest.created_at.desc())
    )

    if status != "all":
        query = query.where(PaymentRequest.status == status)

    results = db.execute(query.limit(100)).all()

    requests = []

    for payment_request, user in results:

        requests.append({
            "request_id": payment_request.id,
            "user_id": user.id,
            "username": user.username,
            "depositor_name": payment_request.depositor_name,
            "amount": payment_request.amount,
            "status": payment_request.status,
            "created_at": payment_request.created_at.isoformat(),
            "reviewed_at": (
                payment_request.reviewed_at.isoformat()
                if payment_request.reviewed_at else None
            ),
        })

    return {
        "success": True,
        "requests": requests,
    }


# =====================================
# 15. 관리자 API - 승인
# =====================================

@app.post("/admin/subscription-requests/{request_id}/approve")
def approve_transfer(
    request_id: int,
    admin: bool = Depends(require_admin),
    db: Session = Depends(get_db),
):

    payment_request = db.get(PaymentRequest, request_id)

    if not payment_request:
        raise HTTPException(
            status_code=404,
            detail="구독 신청을 찾을 수 없습니다.",
        )

    if payment_request.status != "pending":
        raise HTTPException(
            status_code=409,
            detail="이미 처리된 구독 신청입니다.",
        )

    user = db.get(User, payment_request.user_id)

    if not user:
        raise HTTPException(
            status_code=404,
            detail="사용자를 찾을 수 없습니다.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=409,
            detail="정지된 회원은 구독을 승인할 수 없습니다.",
        )

    now = datetime.now(timezone.utc)

    latest = db.scalar(
        select(Subscription)
        .where(Subscription.user_id == user.id)
        .order_by(Subscription.expires_at.desc())
    )

    if latest:
        previous_expiration = make_aware(latest.expires_at)
        started_at = max(now, previous_expiration)
    else:
        started_at = now

    expires_at = started_at + timedelta(days=SUBSCRIPTION_DAYS)

    subscription = Subscription(
        user_id=user.id,
        payment_request_id=payment_request.id,
        started_at=started_at,
        expires_at=expires_at,
    )

    payment_request.status = "approved"
    payment_request.reviewed_at = now

    user.is_premium = True
    user.payment_status = "active"

    db.add(subscription)
    db.commit()

    return {
        "success": True,
        "message": "구독 승인이 완료되었습니다.",
        "username": user.username,
        "started_at": started_at.isoformat(),
        "expires_at": expires_at.isoformat(),
    }


# =====================================
# 16. 관리자 API - 거절
# =====================================

@app.post("/admin/subscription-requests/{request_id}/reject")
def reject_transfer(
    request_id: int,
    admin: bool = Depends(require_admin),
    db: Session = Depends(get_db),
):

    payment_request = db.get(PaymentRequest, request_id)

    if not payment_request:
        raise HTTPException(
            status_code=404,
            detail="구독 신청을 찾을 수 없습니다.",
        )

    if payment_request.status != "pending":
        raise HTTPException(
            status_code=409,
            detail="이미 처리된 구독 신청입니다.",
        )

    payment_request.status = "rejected"
    payment_request.reviewed_at = datetime.now(timezone.utc)

    db.commit()

    return {
        "success": True,
        "message": "구독 신청이 거절되었습니다.",
        "request_id": payment_request.id,
    }


# =====================================
# 17. 관리자 API - 회원 목록
# =====================================

@app.get("/admin/members")
def admin_members(
    search: str = "",
    status: str = "all",
    admin: bool = Depends(require_admin),
    db: Session = Depends(get_db),
):

    if status not in ["all", "premium", "free", "suspended"]:
        raise HTTPException(
            status_code=400,
            detail="올바르지 않은 회원 필터입니다.",
        )

    query = select(User).order_by(User.created_at.desc())

    if search.strip():
        query = query.where(
            User.username.ilike(f"%{search.strip()}%")
        )

    users = db.scalars(query.limit(200)).all()

    members = []

    for user in users:

        state = get_subscription_state(db, user)
        subscription = state["subscription"]

        if status == "premium" and not state["is_premium"]:
            continue

        if status == "free" and state["is_premium"]:
            continue

        if status == "suspended" and user.is_active:
            continue

        members.append({
            "id": user.id,
            "username": user.username,
            "created_at": user.created_at.isoformat(),
            "is_active": user.is_active,
            "is_premium": state["is_premium"],
            "payment_status": state["payment_status"],
            "subscription_expires_at": (
                subscription.expires_at.isoformat()
                if subscription else None
            ),
        })

    total_count = db.scalar(
        select(User.id)
    )

    total_members = len(db.scalars(select(User.id)).all())

    premium_count = 0
    suspended_count = 0

    all_users = db.scalars(select(User)).all()

    for user in all_users:
        if not user.is_active:
            suspended_count += 1
        elif get_subscription_state(db, user)["is_premium"]:
            premium_count += 1

    return {
        "success": True,
        "members": members,
        "total_members": total_members,
        "premium_members": premium_count,
        "free_members": total_members - premium_count - suspended_count,
        "suspended_members": suspended_count,
    }


# =====================================
# 18. 관리자 API - 회원 상세 정보
# =====================================

@app.get("/admin/members/{user_id}")
def admin_member_detail(
    user_id: int,
    admin: bool = Depends(require_admin),
    db: Session = Depends(get_db),
):

    user = db.get(User, user_id)

    if not user:
        raise HTTPException(
            status_code=404,
            detail="회원을 찾을 수 없습니다.",
        )

    state = get_subscription_state(db, user)
    subscription = state["subscription"]

    payments = db.scalars(
        select(PaymentRequest)
        .where(PaymentRequest.user_id == user.id)
        .order_by(PaymentRequest.created_at.desc())
        .limit(50)
    ).all()

    subscriptions = db.scalars(
        select(Subscription)
        .where(Subscription.user_id == user.id)
        .order_by(Subscription.created_at.desc())
        .limit(50)
    ).all()

    return {
        "success": True,
        "member": {
            "id": user.id,
            "username": user.username,
            "created_at": user.created_at.isoformat(),
            "is_active": user.is_active,
            "is_premium": state["is_premium"],
            "payment_status": state["payment_status"],
            "subscription_expires_at": (
                subscription.expires_at.isoformat()
                if subscription else None
            ),
        },
        "payments": [
            {
                "request_id": item.id,
                "depositor_name": item.depositor_name,
                "amount": item.amount,
                "status": item.status,
                "created_at": item.created_at.isoformat(),
                "reviewed_at": (
                    item.reviewed_at.isoformat()
                    if item.reviewed_at else None
                ),
            }
            for item in payments
        ],
        "subscriptions": [
            {
                "subscription_id": item.id,
                "started_at": item.started_at.isoformat(),
                "expires_at": item.expires_at.isoformat(),
                "created_at": item.created_at.isoformat(),
            }
            for item in subscriptions
        ],
    }


# =====================================
# 19. 관리자 API - 수동 구독 지급
# =====================================

@app.post("/admin/members/{user_id}/subscription/grant")
def admin_grant_subscription(
    user_id: int,
    request: AdminGrantRequest,
    admin: bool = Depends(require_admin),
    db: Session = Depends(get_db),
):

    user = db.get(User, user_id)

    if not user:
        raise HTTPException(
            status_code=404,
            detail="회원을 찾을 수 없습니다.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=409,
            detail="정지된 회원에게는 구독을 지급할 수 없습니다.",
        )

    now = datetime.now(timezone.utc)

    latest = db.scalar(
        select(Subscription)
        .where(Subscription.user_id == user.id)
        .order_by(Subscription.expires_at.desc())
    )

    if latest:
        started_at = max(now, make_aware(latest.expires_at))
    else:
        started_at = now

    expires_at = started_at + timedelta(days=request.days)

    # 관리자 수동 지급도 결제/구독 이력에 기록합니다.
    payment_request = PaymentRequest(
        user_id=user.id,
        depositor_name="관리자 수동 지급",
        amount=0,
        status="approved",
        created_at=now,
        reviewed_at=now,
    )

    db.add(payment_request)
    db.flush()

    subscription = Subscription(
        user_id=user.id,
        payment_request_id=payment_request.id,
        started_at=started_at,
        expires_at=expires_at,
    )

    db.add(subscription)

    user.is_premium = True
    user.payment_status = "active"

    db.commit()

    return {
        "success": True,
        "message": "회원에게 구독을 지급했습니다.",
        "username": user.username,
        "days": request.days,
        "started_at": started_at.isoformat(),
        "expires_at": expires_at.isoformat(),
    }


# =====================================
# 20. 관리자 API - 구독 회수
# =====================================

@app.post("/admin/members/{user_id}/subscription/revoke")
def admin_revoke_subscription(
    user_id: int,
    admin: bool = Depends(require_admin),
    db: Session = Depends(get_db),
):

    user = db.get(User, user_id)

    if not user:
        raise HTTPException(
            status_code=404,
            detail="회원을 찾을 수 없습니다.",
        )

    now = datetime.now(timezone.utc)

    subscriptions = db.scalars(
        select(Subscription)
        .where(
            Subscription.user_id == user.id,
            Subscription.expires_at > now,
        )
    ).all()

    for subscription in subscriptions:
        subscription.expires_at = now

    user.is_premium = False
    user.payment_status = "expired"

    db.commit()

    return {
        "success": True,
        "message": "회원의 활성 구독을 회수했습니다.",
        "username": user.username,
    }


# =====================================
# 21. 관리자 API - 계정 정지
# =====================================

@app.post("/admin/members/{user_id}/suspend")
def admin_suspend_member(
    user_id: int,
    admin: bool = Depends(require_admin),
    db: Session = Depends(get_db),
):

    user = db.get(User, user_id)

    if not user:
        raise HTTPException(
            status_code=404,
            detail="회원을 찾을 수 없습니다.",
        )

    user.is_active = False
    db.commit()

    return {
        "success": True,
        "message": "회원 계정을 정지했습니다.",
        "username": user.username,
    }


# =====================================
# 22. 관리자 API - 계정 정지 해제
# =====================================

@app.post("/admin/members/{user_id}/activate")
def admin_activate_member(
    user_id: int,
    admin: bool = Depends(require_admin),
    db: Session = Depends(get_db),
):

    user = db.get(User, user_id)

    if not user:
        raise HTTPException(
            status_code=404,
            detail="회원을 찾을 수 없습니다.",
        )

    user.is_active = True
    db.commit()

    return {
        "success": True,
        "message": "회원 계정 정지가 해제되었습니다.",
        "username": user.username,
    }


# =====================================
# 23. 관리자 홈페이지
# =====================================

ADMIN_HTML = r"""
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EKKO AI 관리자 센터</title>

<style>
* { box-sizing: border-box; }

body {
    margin: 0;
    background: #090912;
    color: #f4f0ff;
    font-family: Arial, "Malgun Gothic", sans-serif;
}

header {
    padding: 22px 35px;
    background: #10101b;
    border-bottom: 1px solid #332544;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.logo {
    color: #a56cff;
    font-size: 23px;
    font-weight: bold;
    letter-spacing: 2px;
}

.tag {
    color: #36e6dc;
    font-size: 12px;
}

main {
    max-width: 1250px;
    margin: 35px auto;
    padding: 0 20px;
}

h1 { font-size: 28px; }
h2 { font-size: 20px; }

.muted { color: #9992aa; }

.panel {
    background: #151522;
    border: 1px solid #332544;
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 20px;
}

input, select {
    width: 100%;
    padding: 13px;
    border-radius: 8px;
    border: 1px solid #51406d;
    background: #0e0d18;
    color: white;
    margin: 8px 0 15px;
}

button {
    border: 0;
    border-radius: 8px;
    padding: 11px 16px;
    color: white;
    font-weight: bold;
    cursor: pointer;
    background: #7544b6;
}

button:hover { opacity: .85; }

button:disabled {
    opacity: .5;
    cursor: not-allowed;
}

.secondary { background: #30243f; }
.approve { background: #167d68; }
.reject { background: #9c3654; }
.warning { background: #8b5b25; }

.cards {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 15px;
    margin-bottom: 20px;
}

.stat {
    background: #151522;
    border: 1px solid #332544;
    border-radius: 12px;
    padding: 20px;
}

.stat small { color: #9992aa; }

.stat strong {
    display: block;
    font-size: 28px;
    margin-top: 9px;
}

.table-wrap { overflow-x: auto; }

table {
    width: 100%;
    border-collapse: collapse;
    min-width: 760px;
}

th, td {
    text-align: left;
    padding: 14px 10px;
    border-bottom: 1px solid #332544;
    font-size: 13px;
}

th { color: #a99bbd; }

.status {
    display: inline-block;
    padding: 5px 9px;
    border-radius: 20px;
    background: #493b24;
    color: #ffd47b;
    font-size: 11px;
}

.status.approved {
    background: #173f35;
    color: #69e6bc;
}

.status.rejected {
    background: #492332;
    color: #ff9bb3;
}

.status.active {
    background: #173f35;
    color: #69e6bc;
}

.status.suspended {
    background: #492332;
    color: #ff9bb3;
}

.actions {
    display: flex;
    gap: 7px;
    flex-wrap: wrap;
}

.hidden { display: none !important; }

#message, #memberMessage {
    margin-top: 12px;
    color: #36e6dc;
    white-space: pre-wrap;
}

.nav {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin: 20px 0;
}

.nav button {
    background: #30243f;
    padding: 13px 20px;
}

.nav button.selected {
    background: #7544b6;
    outline: 1px solid #b58aff;
}

.detail-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
    margin: 15px 0;
}

.detail-item {
    background: #0e0d18;
    border: 1px solid #332544;
    padding: 14px;
    border-radius: 8px;
    overflow-wrap: anywhere;
}

.detail-item small {
    display: block;
    color: #9992aa;
    margin-bottom: 7px;
}

.detail-item strong {
    font-size: 14px;
}

.section-heading {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}

@media(max-width: 800px) {
    .cards { grid-template-columns: repeat(2, 1fr); }
}

@media(max-width: 600px) {
    header { padding: 18px; }
    main { margin: 20px auto; }
    .cards { grid-template-columns: 1fr 1fr; }
    .panel { padding: 17px; }
    .detail-grid { grid-template-columns: 1fr; }
}
</style>
</head>

<body>

<header>
    <div class="logo">EKKO AI ADMIN</div>
    <div class="tag">관리자 전용 시스템</div>
</header>

<main>

<section id="loginPanel" class="panel">
    <h1>관리자 로그인</h1>
    <p class="muted">관리자 인증 키를 입력해 주세요.</p>

    <label>관리자 비밀키</label>
    <input
        id="adminKey"
        type="password"
        placeholder="ADMIN_API_KEY"
        autocomplete="off"
    >

    <button onclick="login()">로그인</button>
    <div id="loginMessage"></div>
</section>

<section id="dashboard" class="hidden">

    <h1>에코 AI 관리자 센터</h1>
    <p class="muted">회원과 구독, 입금 신청을 관리합니다.</p>

    <div class="cards">
        <div class="stat">
            <small>전체 회원</small>
            <strong id="totalMembers">0</strong>
        </div>

        <div class="stat">
            <small>유료 회원</small>
            <strong id="premiumMembers">0</strong>
        </div>

        <div class="stat">
            <small>무료 회원</small>
            <strong id="freeMembers">0</strong>
        </div>

        <div class="stat">
            <small>정지 회원</small>
            <strong id="suspendedMembers">0</strong>
        </div>
    </div>

    <div class="nav">
        <button id="membersTab" class="selected" onclick="showTab('members')">
            회원 관리
        </button>

        <button id="paymentsTab" onclick="showTab('payments')">
            결제 관리
        </button>
    </div>

    <!-- 회원 관리 -->

    <section id="membersSection" class="panel">

        <div class="section-heading">
            <h2>회원 목록</h2>
            <button class="secondary" onclick="loadMembers()">회원 새로고침</button>
        </div>

        <label>회원 검색</label>
        <input
            id="memberSearch"
            placeholder="회원 아이디 검색"
            onkeydown="if(event.key === 'Enter') loadMembers()"
        >

        <label>회원 유형</label>
        <select id="memberFilter" onchange="loadMembers()">
            <option value="all">전체 회원</option>
            <option value="premium">유료 회원</option>
            <option value="free">무료 회원</option>
            <option value="suspended">정지 회원</option>
        </select>

        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>회원 번호</th>
                        <th>아이디</th>
                        <th>가입일</th>
                        <th>회원 상태</th>
                        <th>구독 만료일</th>
                        <th>관리</th>
                    </tr>
                </thead>

                <tbody id="memberRows">
                    <tr>
                        <td colspan="6">회원 정보를 불러오는 중...</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div id="memberMessage"></div>
    </section>

    <!-- 회원 상세 -->

    <section id="memberDetailSection" class="panel hidden">

        <div class="section-heading">
            <h2>회원 상세 정보</h2>
            <button class="secondary" onclick="closeMemberDetail()">닫기</button>
        </div>

        <div id="memberDetail"></div>

        <h2>회원 관리 작업</h2>

        <div class="actions">
            <button class="approve" onclick="grantSubscription()">
                구독 지급
            </button>

            <button class="warning" onclick="revokeSubscription()">
                구독 회수
            </button>

            <button id="suspendButton" class="reject" onclick="toggleSuspension()">
                계정 정지
            </button>
        </div>

        <h2>결제 신청 내역</h2>

        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>신청 번호</th>
                        <th>입금자명</th>
                        <th>금액</th>
                        <th>상태</th>
                        <th>신청일</th>
                    </tr>
                </thead>

                <tbody id="memberPaymentRows"></tbody>
            </table>
        </div>

        <h2>구독 이력</h2>

        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>구독 번호</th>
                        <th>시작일</th>
                        <th>만료일</th>
                    </tr>
                </thead>

                <tbody id="memberSubscriptionRows"></tbody>
            </table>
        </div>
    </section>

    <!-- 결제 관리 -->

    <section id="paymentsSection" class="panel hidden">

        <div class="section-heading">
            <h2>입금 신청 목록</h2>
            <button class="secondary" onclick="loadRequests()">새로고침</button>
        </div>

        <label>신청 상태 필터</label>

        <select id="statusFilter" onchange="loadRequests()">
            <option value="pending">승인 대기</option>
            <option value="approved">승인 완료</option>
            <option value="rejected">거절</option>
            <option value="all">전체 신청</option>
        </select>

        <div class="cards">
            <div class="stat">
                <small>조회 신청</small>
                <strong id="totalCount">0</strong>
            </div>

            <div class="stat">
                <small>승인 대기</small>
                <strong id="pendingCount">0</strong>
            </div>

            <div class="stat">
                <small>승인 완료</small>
                <strong id="approvedCount">0</strong>
            </div>

            <div class="stat">
                <small>거절</small>
                <strong id="rejectedCount">0</strong>
            </div>
        </div>

        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>신청 번호</th>
                        <th>회원 번호</th>
                        <th>회원 아이디</th>
                        <th>입금자명</th>
                        <th>금액</th>
                        <th>신청 일시</th>
                        <th>상태</th>
                        <th>관리</th>
                    </tr>
                </thead>

                <tbody id="requestRows">
                    <tr>
                        <td colspan="8">신청 내역을 불러오는 중...</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div id="message"></div>
    </section>

    <button class="secondary" onclick="logout()">관리자 로그아웃</button>

</section>

</main>

<script>
let adminKey = "";
let selectedMemberId = null;
let selectedMember = null;

const loginPanel = document.getElementById("loginPanel");
const dashboard = document.getElementById("dashboard");

function showMessage(message) {
    document.getElementById("message").textContent = message;
}

function showMemberMessage(message) {
    document.getElementById("memberMessage").textContent = message;
}

async function adminRequest(path, method = "GET", body = null) {

    const options = {
        method: method,
        headers: {
            "X-Admin-Key": adminKey,
            "Accept": "application/json"
        }
    };

    if (body !== null) {
        options.headers["Content-Type"] = "application/json";
        options.body = JSON.stringify(body);
    }

    const response = await fetch(path, options);

    let data = {};

    try {
        data = await response.json();
    } catch (e) {
        throw new Error("서버 응답을 읽을 수 없습니다.");
    }

    if (!response.ok) {
        throw new Error(
            data.detail || "요청에 실패했습니다. HTTP " + response.status
        );
    }

    return data;
}

async function login() {

    const key = document.getElementById("adminKey").value.trim();

    if (!key) {
        document.getElementById("loginMessage").textContent =
            "관리자 비밀키를 입력해 주세요.";
        return;
    }

    adminKey = key;

    try {

        await adminRequest(
            "/admin/subscription-requests?status=pending"
        );

        loginPanel.classList.add("hidden");
        dashboard.classList.remove("hidden");

        await loadMembers();
        await loadRequests();

    } catch (error) {

        adminKey = "";

        document.getElementById("loginMessage").textContent =
            error.message;
    }
}

function logout() {

    adminKey = "";
    selectedMemberId = null;
    selectedMember = null;

    document.getElementById("adminKey").value = "";

    dashboard.classList.add("hidden");
    loginPanel.classList.remove("hidden");

    document.getElementById("loginMessage").textContent =
        "로그아웃되었습니다.";
}

function showTab(tab) {

    const membersSection = document.getElementById("membersSection");
    const paymentsSection = document.getElementById("paymentsSection");
    const detailSection = document.getElementById("memberDetailSection");

    membersSection.classList.toggle("hidden", tab !== "members");
    paymentsSection.classList.toggle("hidden", tab !== "payments");

    if (tab !== "members") {
        detailSection.classList.add("hidden");
    }

    document.getElementById("membersTab").classList.toggle(
        "selected", tab === "members"
    );

    document.getElementById("paymentsTab").classList.toggle(
        "selected", tab === "payments"
    );
}

function formatDate(value) {

    if (!value) return "-";

    const date = new Date(value);

    if (Number.isNaN(date.getTime())) return value;

    return date.toLocaleString("ko-KR");
}

function statusLabel(status) {

    if (status === "approved") return "승인 완료";
    if (status === "rejected") return "거절";
    if (status === "active") return "유료 회원";
    if (status === "expired") return "구독 만료";
    if (status === "pending") return "승인 대기";
    if (status === "suspended") return "계정 정지";

    return "무료 회원";
}

function makeCell(row, value) {

    const cell = document.createElement("td");
    cell.textContent = value ?? "-";
    row.appendChild(cell);

    return cell;
}


// =====================================
// 회원 목록
// =====================================

async function loadMembers() {

    showMemberMessage("회원 정보를 불러오는 중...");

    const search = document.getElementById("memberSearch").value.trim();
    const status = document.getElementById("memberFilter").value;

    try {

        const data = await adminRequest(
            "/admin/members?search=" + encodeURIComponent(search) +
            "&status=" + encodeURIComponent(status)
        );

        document.getElementById("totalMembers").textContent =
            data.total_members;

        document.getElementById("premiumMembers").textContent =
            data.premium_members;

        document.getElementById("freeMembers").textContent =
            data.free_members;

        document.getElementById("suspendedMembers").textContent =
            data.suspended_members;

        const tbody = document.getElementById("memberRows");
        tbody.replaceChildren();

        const members = data.members || [];

        if (members.length === 0) {

            const row = document.createElement("tr");
            const cell = document.createElement("td");

            cell.colSpan = 6;
            cell.textContent = "조회된 회원이 없습니다.";

            row.appendChild(cell);
            tbody.appendChild(row);

        } else {

            for (const member of members) {

                const row = document.createElement("tr");

                makeCell(row, "#" + member.id);
                makeCell(row, member.username);
                makeCell(row, formatDate(member.created_at));

                const statusCell = document.createElement("td");
                const badge = document.createElement("span");

                let label = member.is_active
                    ? (member.is_premium ? "유료 회원" : "무료 회원")
                    : "계정 정지";

                badge.className = "status " +
                    (member.is_active
                        ? (member.is_premium ? "active" : "")
                        : "suspended");

                badge.textContent = label;
                statusCell.appendChild(badge);
                row.appendChild(statusCell);

                makeCell(
                    row,
                    formatDate(member.subscription_expires_at)
                );

                const actionCell = document.createElement("td");
                const detailButton = document.createElement("button");

                detailButton.textContent = "상세 관리";
                detailButton.className = "secondary";

                detailButton.addEventListener("click", () => {
                    openMemberDetail(member.id);
                });

                actionCell.appendChild(detailButton);
                row.appendChild(actionCell);

                tbody.appendChild(row);
            }
        }

        showMemberMessage(
            "회원 목록을 업데이트했습니다. 최대 200명까지 표시됩니다."
        );

    } catch (error) {

        showMemberMessage(error.message);
    }
}


// =====================================
// 회원 상세 정보
// =====================================

async function openMemberDetail(userId) {

    try {

        const data = await adminRequest(
            "/admin/members/" + userId
        );

        selectedMemberId = userId;
        selectedMember = data.member;

        const member = data.member;

        const detail = document.getElementById("memberDetail");
        detail.replaceChildren();

        const grid = document.createElement("div");
        grid.className = "detail-grid";

        const fields = [
            ["회원 번호", "#" + member.id],
            ["아이디", member.username],
            ["가입일", formatDate(member.created_at)],
            ["계정 상태", member.is_active ? "정상" : "정지"],
            ["구독 상태", member.is_premium ? "유료 회원" : "무료 회원"],
            ["구독 만료일", formatDate(member.subscription_expires_at)]
        ];

        for (const field of fields) {

            const item = document.createElement("div");
            item.className = "detail-item";

            const small = document.createElement("small");
            small.textContent = field[0];

            const strong = document.createElement("strong");
            strong.textContent = field[1];

            item.appendChild(small);
            item.appendChild(strong);
            grid.appendChild(item);
        }

        detail.appendChild(grid);

        const suspendButton = document.getElementById("suspendButton");

        if (member.is_active) {
            suspendButton.textContent = "계정 정지";
            suspendButton.className = "reject";
        } else {
            suspendButton.textContent = "정지 해제";
            suspendButton.className = "approve";
        }

        renderPaymentHistory(data.payments || []);
        renderSubscriptionHistory(data.subscriptions || []);

        document.getElementById(
            "memberDetailSection"
        ).classList.remove("hidden");

        showMemberMessage(
            member.username + " 회원 정보를 불러왔습니다."
        );

        document.getElementById("memberDetailSection").scrollIntoView({
            behavior: "smooth",
            block: "start"
        });

    } catch (error) {

        showMemberMessage(error.message);
    }
}

function renderPaymentHistory(payments) {

    const tbody = document.getElementById("memberPaymentRows");
    tbody.replaceChildren();

    if (payments.length === 0) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");

        cell.colSpan = 5;
        cell.textContent = "결제 신청 내역이 없습니다.";

        row.appendChild(cell);
        tbody.appendChild(row);
        return;
    }

    for (const payment of payments) {

        const row = document.createElement("tr");

        makeCell(row, "#" + payment.request_id);
        makeCell(row, payment.depositor_name);
        makeCell(row, Number(payment.amount).toLocaleString("ko-KR") + "원");
        makeCell(row, statusLabel(payment.status));
        makeCell(row, formatDate(payment.created_at));

        tbody.appendChild(row);
    }
}

function renderSubscriptionHistory(subscriptions) {

    const tbody = document.getElementById("memberSubscriptionRows");
    tbody.replaceChildren();

    if (subscriptions.length === 0) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");

        cell.colSpan = 3;
        cell.textContent = "구독 이력이 없습니다.";

        row.appendChild(cell);
        tbody.appendChild(row);
        return;
    }

    for (const subscription of subscriptions) {

        const row = document.createElement("tr");

        makeCell(row, "#" + subscription.subscription_id);
        makeCell(row, formatDate(subscription.started_at));
        makeCell(row, formatDate(subscription.expires_at));

        tbody.appendChild(row);
    }
}

function closeMemberDetail() {

    document.getElementById(
        "memberDetailSection"
    ).classList.add("hidden");

    selectedMemberId = null;
    selectedMember = null;
}


// =====================================
// 구독 수동 지급
// =====================================

async function grantSubscription() {

    if (!selectedMemberId || !selectedMember) return;

    if (!selectedMember.is_active) {
        alert("정지된 회원에게는 구독을 지급할 수 없습니다.");
        return;
    }

    const daysText = prompt(
        "지급할 구독 기간을 입력해 주세요.\n기본값은 30일입니다.",
        "30"
    );

    if (daysText === null) return;

    const days = Number(daysText);

    if (!Number.isInteger(days) || days < 1 || days > 3650) {
        alert("1일에서 3650일 사이의 정수로 입력해 주세요.");
        return;
    }

    if (!confirm(
        selectedMember.username + " 회원에게 " +
        days + "일 구독을 지급할까요?"
    )) return;

    try {

        const data = await adminRequest(
            "/admin/members/" + selectedMemberId +
            "/subscription/grant",
            "POST",
            { days: days }
        );

        alert(
            data.message + "\n만료일: " +
            formatDate(data.expires_at)
        );

        await loadMembers();
        await openMemberDetail(selectedMemberId);

    } catch (error) {

        alert(error.message);
    }
}


// =====================================
// 구독 회수
// =====================================

async function revokeSubscription() {

    if (!selectedMemberId || !selectedMember) return;

    if (!confirm(
        selectedMember.username +
        " 회원의 현재 활성 구독을 즉시 회수할까요?"
    )) return;

    try {

        const data = await adminRequest(
            "/admin/members/" + selectedMemberId +
            "/subscription/revoke",
            "POST"
        );

        alert(data.message);

        await loadMembers();
        await openMemberDetail(selectedMemberId);

    } catch (error) {

        alert(error.message);
    }
}


// =====================================
// 계정 정지 및 해제
// =====================================

async function toggleSuspension() {

    if (!selectedMemberId || !selectedMember) return;

    const isActive = selectedMember.is_active;

    const action = isActive ? "정지" : "정지 해제";

    if (!confirm(
        selectedMember.username +
        " 회원의 계정을 " + action + "할까요?"
    )) return;

    try {

        const endpoint = isActive ? "suspend" : "activate";

        const data = await adminRequest(
            "/admin/members/" + selectedMemberId + "/" + endpoint,
            "POST"
        );

        alert(data.message);

        await loadMembers();
        await openMemberDetail(selectedMemberId);

    } catch (error) {

        alert(error.message);
    }
}


// =====================================
// 결제 신청 목록
// =====================================

async function loadRequests() {

    showMessage("신청 내역을 불러오는 중...");

    const status = document.getElementById("statusFilter").value;

    try {

        const data = await adminRequest(
            "/admin/subscription-requests?status=" +
            encodeURIComponent(status)
        );

        const requests = data.requests || [];

        document.getElementById("totalCount").textContent =
            requests.length;

        document.getElementById("pendingCount").textContent =
            requests.filter(item => item.status === "pending").length;

        document.getElementById("approvedCount").textContent =
            requests.filter(item => item.status === "approved").length;

        document.getElementById("rejectedCount").textContent =
            requests.filter(item => item.status === "rejected").length;

        const tbody = document.getElementById("requestRows");
        tbody.replaceChildren();

        if (requests.length === 0) {

            const row = document.createElement("tr");
            const cell = document.createElement("td");

            cell.colSpan = 8;
            cell.textContent = "표시할 신청 내역이 없습니다.";

            row.appendChild(cell);
            tbody.appendChild(row);

        } else {

            for (const item of requests) {

                const row = document.createElement("tr");

                makeCell(row, "#" + item.request_id);
                makeCell(row, "#" + item.user_id);
                makeCell(row, item.username);
                makeCell(row, item.depositor_name);
                makeCell(
                    row,
                    Number(item.amount).toLocaleString("ko-KR") + "원"
                );
                makeCell(row, formatDate(item.created_at));

                const statusCell = document.createElement("td");
                const badge = document.createElement("span");

                badge.className = "status " + item.status;
                badge.textContent = statusLabel(item.status);

                statusCell.appendChild(badge);
                row.appendChild(statusCell);

                const actionCell = document.createElement("td");

                if (item.status === "pending") {

                    const actions = document.createElement("div");
                    actions.className = "actions";

                    const approveButton = document.createElement("button");
                    approveButton.className = "approve";
                    approveButton.textContent = "승인";

                    approveButton.addEventListener("click", () => {
                        approveRequest(item.request_id, item.username);
                    });

                    const rejectButton = document.createElement("button");
                    rejectButton.className = "reject";
                    rejectButton.textContent = "거절";

                    rejectButton.addEventListener("click", () => {
                        rejectRequest(item.request_id, item.username);
                    });

                    actions.appendChild(approveButton);
                    actions.appendChild(rejectButton);
                    actionCell.appendChild(actions);

                } else {

                    actionCell.textContent = "-";
                }

                row.appendChild(actionCell);
                tbody.appendChild(row);
            }
        }

        showMessage("신청 내역을 업데이트했습니다.");

    } catch (error) {

        showMessage(error.message);
    }
}


// =====================================
// 결제 승인
// =====================================

async function approveRequest(id, username) {

    const confirmed = confirm(
        "회원 아이디: " + username +
        "\n신청 번호: " + id +
        "\n\n실제 입금 내역을 확인했나요? 승인할까요?"
    );

    if (!confirmed) return;

    try {

        const data = await adminRequest(
            "/admin/subscription-requests/" + id + "/approve",
            "POST"
        );

        showMessage(
            data.message + "\n회원: " + data.username +
            "\n만료일: " + formatDate(data.expires_at)
        );

        await loadRequests();
        await loadMembers();

    } catch (error) {

        showMessage(error.message);
    }
}


// =====================================
// 결제 거절
// =====================================

async function rejectRequest(id, username) {

    const confirmed = confirm(
        "회원 아이디: " + username +
        "\n신청 번호: " + id +
        "\n\n이 신청을 거절할까요?"
    );

    if (!confirmed) return;

    try {

        const data = await adminRequest(
            "/admin/subscription-requests/" + id + "/reject",
            "POST"
        );

        showMessage(data.message);

        await loadRequests();
        await loadMembers();

    } catch (error) {

        showMessage(error.message);
    }
}


document.getElementById("adminKey").addEventListener(
    "keydown",
    function(event) {
        if (event.key === "Enter") login();
    }
);

</script>

</body>
</html>
"""


@app.get("/admin", response_class=HTMLResponse)
def admin_homepage():
    return HTMLResponse(
        content=ADMIN_HTML,
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


# =====================================
# 24. AI 채팅
# =====================================

@app.post("/chat")
def chat(
    request: ChatRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):

    require_active_subscription(user, db)

    try:

        client = OpenAI(api_key=OPENAI_API_KEY)

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
