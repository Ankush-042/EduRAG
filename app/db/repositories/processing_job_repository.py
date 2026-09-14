"""Persistence for processing_jobs — the state-machine bookkeeping the UI
reads to show real per-stage progress (UI/UX spec Doc 3 sec 10-11)."""

from sqlalchemy.orm import Session as DbSession

from app.db.base import _utcnow
from app.db.models.processing import ProcessingJob


def create_job(db: DbSession, *, source_id: str, job_type: str) -> ProcessingJob:
    job = ProcessingJob(source_id=source_id, job_type=job_type, status="PENDING", progress=0.0)
    db.add(job)
    db.flush()
    return job


def get_job(db: DbSession, job_id: str) -> ProcessingJob | None:
    """Sprint 7: looked up by id from the background pipeline thread, which
    has its own DB session and so can't hold onto the ProcessingJob object
    the request thread created."""
    return db.get(ProcessingJob, job_id)


def start_job(db: DbSession, job: ProcessingJob, *, stage: str) -> ProcessingJob:
    job.status = "RUNNING"
    job.current_stage = stage
    job.started_at = _utcnow()
    db.flush()
    return job


def update_progress(db: DbSession, job: ProcessingJob, *, progress: float, stage: str) -> ProcessingJob:
    job.progress = progress
    job.current_stage = stage
    db.flush()
    return job


def complete_job(db: DbSession, job: ProcessingJob) -> ProcessingJob:
    job.status = "COMPLETED"
    job.progress = 1.0
    job.completed_at = _utcnow()
    db.flush()
    return job


def fail_job(db: DbSession, job: ProcessingJob, *, error_message: str) -> ProcessingJob:
    job.status = "FAILED"
    job.error_message = error_message
    job.completed_at = _utcnow()
    db.flush()
    return job


def latest_job_for_source(db: DbSession, source_id: str) -> ProcessingJob | None:
    return (
        db.query(ProcessingJob)
        .filter(ProcessingJob.source_id == source_id)
        .order_by(ProcessingJob.started_at.desc().nullslast())
        .first()
    )
