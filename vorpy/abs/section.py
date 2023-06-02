import abc
import numpy.typing as npt


class Section(abc.ABC):

    @abc.abstractmethod
    def cl(self, aoa: npt.NDArray) -> npt.NDArray:
        """Lift coefficient as function of attack angle"""
        ...
    
    @abc.abstractmethod
    def cd(self, aoa: npt.NDArray) -> npt.NDArray:
        """Drag coefficient as function of attack angle"""
        ...
    
    @abc.abstractmethod
    def cm(self, aoa: npt.NDArray) -> npt.NDArray:
        """Moment coefficient as function of attack angle"""
        ...

    @abc.abstractmethod
    def update(self, time: float) -> None:
        """Update the properties below based on the simulation time"""
        ...

    @abc.abstractproperty
    def origin(self) -> npt.NDArray:
        """Airfoil 1/4 chord position"""
        ...

    @abc.abstractproperty
    def e1(self) -> npt.NDArray:
        """Base system: x axis"""
        ...
    
    @abc.abstractproperty
    def e2(self) -> npt.NDArray:
        """Base system: y axis"""
        ...
    
    @abc.abstractproperty
    def e3(self) -> npt.NDArray:
        """Base system: z axis"""
        ...
    
    @abc.abstractproperty
    def chord(self) -> float:
        """Airfoil chord"""
        ...
    
    @abc.abstractproperty
    def linear_velocity(self) -> npt.NDArray:
        """Origin velocity"""
        ...
    
    @abc.abstractproperty
    def linear_acceleration(self) -> npt.NDArray:
        """Origin acceleration"""
        ...
    
    @abc.abstractproperty
    def angular_velocity(self) -> npt.NDArray:
        """Section angular velocity"""
        ...
    
    @abc.abstractproperty
    def angular_acceleration(self) -> npt.NDArray:
        """Section angular acceleration"""
        ...
    