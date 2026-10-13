"""SQLAlchemy ORM models -- the PostgreSQL schema (spec #48).

Includes documents, document_versions, chunks, ingestion_jobs,
index_versions, evaluation_datasets, evaluation_questions, evaluation_runs,
evaluation_results, query_logs, users, groups, document_permissions.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    source_uri: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str | None] = mapped_column(Text)
    author: Mapped[str | None] = mapped_column(String(256))
    repository: Mapped[str | None] = mapped_column(String(256))
    branch: Mapped[str | None] = mapped_column(String(128))
    commit_sha: Mapped[str | None] = mapped_column(String(64))
    url: Mapped[str | None] = mapped_column(Text)
    file_path: Mapped[str | None] = mapped_column(Text)
    language: Mapped[str | None] = mapped_column(String(32))
    mime_type: Mapped[str | None] = mapped_column(String(128))
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    current_version: Mapped[int] = mapped_column(Integer, default=1)
    acl_tenant_id: Mapped[str | None] = mapped_column(String(64), index=True)
    acl_visibility: Mapped[str] = mapped_column(String(16), default="public")
    doc_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))  # soft delete

    versions: Mapped[list[DocumentVersion]] = relationship(back_populates="document", cascade="all, delete-orphan")
    chunks: Mapped[list[Chunk]] = relationship(back_populates="document", cascade="all, delete-orphan")
    permissions: Mapped[list[DocumentPermission]] = relationship(back_populates="document", cascade="all, delete-orphan")

    __table_args__ = (Index("ix_documents_source_uri_hash", "source_uri", "content_hash"),)


class DocumentVersion(Base):
    __tablename__ = "document_versions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(String(64), ForeignKey("documents.id"), nullable=False)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    parser_version: Mapped[str] = mapped_column(String(32), nullable=False)
    chunking_strategy: Mapped[str] = mapped_column(String(32), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(128), nullable=False)
    embedding_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    document: Mapped[Document] = relationship(back_populates="versions")

    __table_args__ = (UniqueConstraint("document_id", "version_number", name="uq_doc_version"),)


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    document_id: Mapped[str] = mapped_column(String(64), ForeignKey("documents.id"), nullable=False, index=True)
    parent_chunk_id: Mapped[str | None] = mapped_column(String(128))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    character_count: Mapped[int] = mapped_column(Integer, nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    section: Mapped[str | None] = mapped_column(Text)
    heading_path: Mapped[list] = mapped_column(JSON, default=list)
    language: Mapped[str | None] = mapped_column(String(32))
    chunk_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    embedding: Mapped[list | None] = mapped_column(JSON)  # list[float]; pgvector column added via migration for prod
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (Index("ix_chunks_document_position", "document_id", "position"),)


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    source_uri: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="queued")  # queued|running|completed|failed|retrying
    document_id: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class IndexVersion(Base):
    __tablename__ = "index_versions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    embedding_model: Mapped[str] = mapped_column(String(128), nullable=False)
    embedding_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    bm25_k1: Mapped[float] = mapped_column(Float, nullable=False)
    bm25_b: Mapped[float] = mapped_column(Float, nullable=False)
    ann_backend: Mapped[str] = mapped_column(String(32), nullable=False)
    document_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EvaluationDataset(Base):
    __tablename__ = "evaluation_datasets"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    question_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    questions: Mapped[list[EvaluationQuestionRow]] = relationship(back_populates="dataset", cascade="all, delete-orphan")


class EvaluationQuestionRow(Base):
    __tablename__ = "evaluation_questions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(String(64), ForeignKey("evaluation_datasets.id"), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    expected_answer: Mapped[str] = mapped_column(Text, nullable=False)
    relevant_documents: Mapped[list] = mapped_column(JSON, default=list)
    relevant_chunks: Mapped[list] = mapped_column(JSON, default=list)
    difficulty: Mapped[str] = mapped_column(String(16), nullable=False)
    topic: Mapped[str] = mapped_column(String(64), nullable=False)
    query_type: Mapped[str] = mapped_column(String(32), nullable=False)

    dataset: Mapped[EvaluationDataset] = relationship(back_populates="questions")


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    dataset_id: Mapped[str] = mapped_column(String(64), ForeignKey("evaluation_datasets.id"), nullable=False)
    experiment_name: Mapped[str] = mapped_column(String(256), nullable=False)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    git_commit: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    results: Mapped[list[EvaluationResult]] = relationship(back_populates="run", cascade="all, delete-orphan")


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(String(64), ForeignKey("evaluation_runs.id"), nullable=False)
    question_id: Mapped[str] = mapped_column(String(64), nullable=False)
    retrieved_chunk_ids: Mapped[list] = mapped_column(JSON, default=list)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    recall_at_10: Mapped[float] = mapped_column(Float, default=0.0)

    run: Mapped[EvaluationRun] = relationship(back_populates="results")


class QueryLog(Base):
    __tablename__ = "query_logs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    request_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    query_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    retrieval_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    user_id: Mapped[str | None] = mapped_column(String(64), index=True)
    tenant_id: Mapped[str | None] = mapped_column(String(64), index=True)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    result_count: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(256), nullable=False, unique=True)
    hashed_password: Mapped[str] = mapped_column(String(256), nullable=False)
    role: Mapped[str] = mapped_column(String(16), default="user")  # user|admin
    tenant_id: Mapped[str | None] = mapped_column(String(64), index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Group(Base):
    __tablename__ = "groups"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    tenant_id: Mapped[str | None] = mapped_column(String(64), index=True)


class DocumentPermission(Base):
    __tablename__ = "document_permissions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(String(64), ForeignKey("documents.id"), nullable=False, index=True)
    principal_type: Mapped[str] = mapped_column(String(16), nullable=False)  # "user" | "group"
    principal_id: Mapped[str] = mapped_column(String(64), nullable=False)
    permission: Mapped[str] = mapped_column(String(16), default="read")

    document: Mapped[Document] = relationship(back_populates="permissions")

    __table_args__ = (
        UniqueConstraint("document_id", "principal_type", "principal_id", name="uq_doc_permission"),
        Index("ix_doc_permissions_principal", "principal_type", "principal_id"),
    )
