"""Database schema and utilities"""
import re
from contextlib import contextmanager
from typing import List, Optional, Tuple

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, declared_attr, mapped_column, relationship, sessionmaker

engine = create_engine("sqlite:///data/database_by_sex.sqlite")
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

    network: Mapped[str] = mapped_column(String(1024))
    """Network name"""
    reduce_by_male: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    """Test metric: Male/Female/Both"""
    reduction: Mapped[int] = mapped_column(Integer)
    """Reduction index"""
    dataset_size: Mapped[float] = mapped_column(Float)
    """Dataset size as fraction from source"""
    history: Mapped[List["TrainHistory"]] = relationship(back_populates="network")
    """Train history relationship"""
    performance: Mapped[List["ClassPerformance"]] = relationship(back_populates="network")
    """Class performance relationship"""
    classification: Mapped[List["ClassificationReport"]] = relationship(back_populates="network")
    """Class performance relationship"""


class ClassPerformance(Base):
    """Storing per class performance on each individual diagnosis"""

    network_id: Mapped[int] = mapped_column(ForeignKey("run_status.id"))
    """Network train information FK"""
    network: Mapped["RunStatus"] = relationship(back_populates="performance")
    """Relationship with network"""
    diagnosis_id: Mapped[Optional[int]] = mapped_column(ForeignKey("diagnosis.id"), nullable=True, default=None)
    """Diagnosis FK. Null mean general performance"""
    diagnosis: Mapped[Optional["Diagnosis"]] = relationship(back_populates="performance")
    """Relationship with diagnosis"""
    test_metric_is_male: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    """Test metric: Male/Female/Both"""
    accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=None)
    """Accuracy on test dataset"""
    loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=None)
    """Loss on test dataset"""


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
    support: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    """Support"""


class TrainHistory(Base):
    """Run status recorder"""

    network_id: Mapped[int] = mapped_column(ForeignKey("run_status.id"))
    """Network train information FK"""
    network: Mapped["RunStatus"] = relationship(back_populates="history")
    """Relationship with network"""

    epoch: Mapped[int] = mapped_column(Integer)
    """Train epoch"""
    train_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=None)
    """Train performance"""
    train_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=None)
    """Train loss"""
    validation_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=None)
    """Validation performance"""
    validation_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=None)
    """Validation loss"""
    learning_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=None)
    """Validation loss"""


class Images(Base):
    """Images info table"""

    filename: Mapped[str] = mapped_column(String(1024), unique=True)
    """Image filename"""
    patient: Mapped[int] = mapped_column(Integer)
    """Patient ID"""
    age: Mapped[int] = mapped_column(Integer)
    """Patient age"""
    is_male: Mapped[bool] = mapped_column(Boolean)
    """Patient sex (set if is male)"""
    is_test: Mapped[bool] = mapped_column(Boolean, default=False)
    """Set if image used for test process"""
    is_validation: Mapped[bool] = mapped_column(Boolean, default=False)
    """Set if image used for validation process"""
    is_train: Mapped[bool] = mapped_column(Boolean, default=False)
    """Set if image used for train process"""
    ignored: Mapped[bool] = mapped_column(Boolean, default=False)
    """Set if image ignored for all operations"""
    reduction_index: Mapped[int] = mapped_column(Integer, default=0)
    """Reduction index"""
    diagnosis_id: Mapped[int] = mapped_column(ForeignKey("diagnosis.id"))
    """Diagnosis FK"""
    diagnosis: Mapped["Diagnosis"] = relationship(back_populates="image")
    """Relationship with original diagnosis"""


class Diagnosis(Base):
    """Image diagnosis"""

    diagnosis: Mapped[str] = mapped_column(String(64))
    """Diagnosis"""
    image: Mapped[List["Images"]] = relationship(back_populates="diagnosis", foreign_keys="Images.diagnosis_id")
    """Relationship image. Original diagnosis"""
    ignored: Mapped[bool] = mapped_column(Boolean, default=False)
    """Set if diagnosis is ignored"""
    performance: Mapped[List["ClassPerformance"]] = relationship(back_populates="diagnosis")
    """Relation ship with class performance"""


def get_diagnosis_list(session) -> Tuple[str, ...]:
    """Return list of existing diagnosis

    Parameters
    ----------
    session: Session
        Active session

    Returns
    -------
    Tuple[str, ...]
        Diagnosis labels alphabetically ordered
    """
    return tuple(
        row.diagnosis
        for row in session.query(Diagnosis.diagnosis)
        .filter(Diagnosis.ignored.is_(False))
        .order_by(Diagnosis.diagnosis.asc())
    )
