
from abc import ABC, abstractmethod
from typing import Any, Dict, Callable

# Type aliases for domain-generic objects used throughout the ORIUS pipeline.
State = Any
Action = Any
UncertaintySet = Any
Config = Dict[str, Any]


class Plant(ABC):
    """
    Abstract base class for a physical system's dynamics model (the "plant").
    """

    @abstractmethod
    def step(self, action: Action) -> State:
        """
        Advances the simulation by one time step based on the given action.
        Returns the new true state of the system.
        """
        pass

    @property
    @abstractmethod
    def state(self) -> State:
        """
        Returns the current true state of the system.
        """
        pass


class Optimizer(ABC):
    """
    Abstract base class for a controller that proposes candidate actions.
    """

    @abstractmethod
    def get_candidate_action(self, state: State, forecast: Any) -> Action:
        """
        Calculates a candidate action based on the observed state and forecasts.
        """
        pass


class DomainAdapter(ABC):
    """
    Abstract base class for a domain-specific adapter.

    This interface bundles all the domain-specific components (physics model,
    controller, safety rules, metrics) required by the generic Orius framework.
    """

    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def get_plant(self) -> Plant:
        """
        Returns an instance of the domain's physical plant model.
        """
        pass

    @abstractmethod
    def get_optimizer(self) -> Optimizer:
        """
        Returns an instance of the domain's candidate action optimizer.
        """
        pass

    @abstractmethod
    def project_to_safe_set(
        self, candidate_action: Action, uncertainty_set: UncertaintySet
    ) -> Action:
        """
        Repairs a candidate action to ensure it is safe.

        This is the core of the domain-specific safety logic. It takes the
        optimizer's proposed action and the current uncertainty about the true
        state, and returns a new action that is guaranteed to be safe.
        """
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Callable[[Any], float]]:
        """
        Returns a dictionary of domain-specific metric functions for evaluation.
        The key is the metric name and the value is a callable that computes it.
        """
        pass

    @abstractmethod
    def get_oqe_features(self, telemetry: Any) -> Dict[str, float]:
        """
        Computes domain-specific features for the Observation Quality Engine (OQE)
        from a raw telemetry event.
        """
        pass

