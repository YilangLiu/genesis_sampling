import torch
from abc import ABC, abstractmethod
from typing import Tuple

class BaseEnv(ABC):

    @abstractmethod
    def __init__(self, num_envs: int, dt:float=0.01, show_viewer=False, show_FPS=False, offscreen_cam=False):
        pass

    @abstractmethod
    def step(self, actions: torch.Tensor):
        pass
    
    @abstractmethod
    def get_states(self) -> torch.Tensor:
        pass
    
    @abstractmethod
    def reset_with_states(self, states:torch.Tensor):
        pass

    @abstractmethod
    def reset(self) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_rewards(self, states:torch.Tensor, actions:torch.Tensor) -> torch.Tensor:
        pass
    
    def set_times(self, ts:torch.Tensor):
        """
        Set the time of each envs with ts:
        Args:
            ts: time to set. Shape: (num_of_envs, ).
        """
        self._t = ts.clone()
    
    def get_times(self) -> torch.Tensor:
        """
        Sim time of each environments
        """
        return self._t 
    
    def get_offscreen_cam(self):
        """
        Get the offscreen camera if there exist one
        """
        return self._offscreen_cam

    @property
    def dt(self) -> float:
        return self._dt
    
    @property
    def num_envs(self) -> int:
        return self._num_envs
    
    @property
    def num_actions(self) -> int:
        return self._num_actions
    
    @property
    def num_states(self) -> int:
        return self._num_states

    @property
    def actions_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._actions_limits
