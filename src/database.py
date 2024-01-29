"""Database schema and utilities"""
import logging
import re
from contextlib import contextmanager
from typing import List, Optional, Union

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import DeclarativeBase, Mapped, declared_attr, mapped_column, relationship, sessionmaker

from config import settings
from history import EvaluationReport, TrainHistory

engine = create_engine("sqlite:///data/database.sqlite")
Session = sessionmaker(engine)


@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    session = Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# pylint: disable=too-few-public-methods
class Base(DeclarativeBase):
    """Base table"""

    @declared_attr
    def __tablename__(cls):  # pylint: disable=no-self-argument
        return re.sub(r"(?<!^)(?=[A-Z])", "_", cls.__name__).lower()

    id: Mapped[int] = mapped_column(primary_key=True)
    """Record ID"""


class RunStatus(Base):
    """Run status recorder"""

    network: Mapped[str] = mapped_column(String(1024))  # pylint: disable=unsubscriptable-object
    """Network name"""
    reduce_by_male: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    """Test metric: Male/Female/Both"""
    corruption: Mapped[int] = mapped_column(Integer)
    """Corruption percentage"""
    reduction: Mapped[int] = mapped_column(Integer)
    """Corruption percentage"""
    epoch: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, default=None)
    """Train epoch"""
    train_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=None)
    """Train performance"""
    validation_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=None)
    """Validation performance"""
    done: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    """Set if evaluation completed"""
    classification: Mapped[List["ClassificationReport"]] = relationship(back_populates="network")
    """Class performance relationship"""


class ClassificationReport(Base):
    """Store classification report"""

    network_id: Mapped[int] = mapped_column(ForeignKey("run_status.id"))
    """Network train information FK"""
    network: Mapped["RunStatus"] = relationship(back_populates="classification")
    """Relationship with network"""
    test_metric_is_male: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    """Test metric: Male/Female/Both"""
    label: Mapped[str] = mapped_column(String(1024), nullable=False)
    """Measured label (class/average)"""
    precision: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    """Precision"""
    recall: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    """Recall"""
    f1_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    """F1 score"""


def create_new_run(
    network: str, corruption: int, reduction: float, reduce_by_male: Union[bool, None]
) -> Union[int, None]:
    """Check if network is evaluated with current settings and if not create new run

    Parameters
    ----------
    network: str
        Network name
    corruption: int
        Corruption percentage
    reduction: int
        Reduction percentage
    reduce_by_male:  Union[bool, None]
        Set apply corruption and reduction to male patients only, Clear to female, None to both

    Returns
    -------
    Union[int, None]
        If network already evaluated return None, otherwise record ID
    """
    logger = logging.getLogger("raido")
    with session_scope() as session:
        try:
            _id = (
                session.query(RunStatus.id, RunStatus.done)
                .filter(RunStatus.network == network)
                .filter(RunStatus.corruption == int(corruption))
                .filter(RunStatus.reduction == int(reduction))
                .filter(RunStatus.reduce_by_male.is_(reduce_by_male))
            ).one()
            if _id.done is True:
                return None
            return _id.id
        except NoResultFound:
            logger.info("No record found creating new")
        _run = RunStatus(network=network, corruption=corruption, reduction=reduction, reduce_by_male=reduce_by_male)
        session.add(_run)
        session.flush()
        session.refresh(_run)
        return _run.id


def update_train_values(run_id: int, history: TrainHistory) -> None:
    """Update measured corruption and reduction values

    Parameters
    ----------
    run_id: int
        Run ID
    history: TrainHistory
        Train results
    """
    with session_scope() as session:
        _record = session.query(RunStatus).filter(RunStatus.id == run_id).one()  # type: RunStatus
        _record.train_accuracy = history.train_accuracy
        _record.validation_accuracy = history.validation_accuracy
        _record.epoch = len(history.accuracy_history[0])


def update_evaluation(run_id: int, evaluation: EvaluationReport) -> None:
    """Save evaluation metrics

    Parameters
    ----------
    run_id: int
        Run ID
    evaluation: EvaluationReport
        Evaluation results
    """
    evaluation.male.f1_score()
    with session_scope() as session:
        for group_name, group in [("male", True), ("female", False), ("common", None)]:
            for diagnosis in settings.preprocessing.target_diagnosis + [
                "macro avg",
            ]:
                record = ClassificationReport(
                    network_id=run_id,
                    test_metric_is_male=group,
                    label=diagnosis,
                    precision=getattr(evaluation, group_name).precision(diagnosis),
                    recall=getattr(evaluation, group_name).recall(diagnosis),
                    f1_score=getattr(evaluation, group_name).f1_score(diagnosis),
                )
                session.add(record)


def mark_evaluation_completed(run_id: int) -> None:
    """Mark evaluation run as completed

    Parameters
    ----------
    run_id: int
        Run ID
    """
    with session_scope() as session:
        _record = session.query(RunStatus).filter(RunStatus.id == run_id).one()  # type: RunStatus
        _record.done = True


if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
