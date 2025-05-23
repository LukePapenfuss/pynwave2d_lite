from abc import ABC, abstractmethod
import numpy as np
from enum import Enum


class FilterType(Enum):
    """
    Enum for filter types
    """

    NONE = 1
    KREISS_OLIGER_O6 = 2
    KREISS_OLIGER_O8 = 3


class FilterApply(Enum):
    """
    Enum for filter application
    """

    NONE = 1
    RHS = 2
    APPLY_VARS = 3
    APPLY_DERIVS = 4


class Filter1D(ABC):
    """
    Abstract base class for a filters
    """

    def __init__(self, dx, apply_filter: FilterApply, filter_type: FilterType):
        self.dx = dx
        self.apply_filter = apply_filter
        self.filter_type = filter_type

    @abstractmethod
    def filter(self, du, u):
        pass

    def get_filter_type(self):
        return self.filter_type

    def get_apply_filter(self):
        return self.apply_filter


class Filter2D(ABC):
    """
    Abstract base class for a filters
    """

    def __init__(self, dx, apply_filter: FilterApply, filter_type: FilterType):
        self.dx = dx
        self.apply_filter = apply_filter
        self.filter_type = filter_type

    @abstractmethod
    def filter(self, u):
        pass

    def get_filter_type(self):
        return self.filter_type

    def get_apply_filter(self):
        return self.apply_filter


class KreissOligerFilterO6_1D(Filter1D):
    """
    Kreiss-Oliger filter in 1D
    """

    def __init__(self, dx, sigma, apply_diss_boundaries=True):
        self.sigma = sigma
        self.APPLY_DISS_BOUNDARIES = apply_diss_boundaries
        apply_filter = FilterApply.RHS
        filter_type = FilterType.KREISS_OLIGER_O6
        super().__init__(dx, apply_filter, filter_type)

    def get_sigma(self):
        return self.sigma

    def filter(self, du, u):
        # Kreiss-Oliger filter in x direction
        dx = self.dx
        sigma = self.sigma
        factor = sigma / (64.0 * dx)

        # centered stencil
        du[3:-3] = factor * (
            u[:-6]
            - 6.0 * u[1:-5]
            + 15.0 * u[2:-4]
            - 20.0 * u[3:-3]
            + 15.0 * u[4:-2]
            - 6.0 * u[5:-1]
            + u[6:]
        )

        if self.APPLY_DISS_BOUNDARIES:
            smr3 = 9.0 / 48.0 * 64 * dx
            smr2 = 43.0 / 48.0 * 64 * dx
            smr1 = 49.0 / 48.0 * 64 * dx
            spr3 = smr3
            spr2 = smr2
            spr1 = smr1
            du[0] = sigma * (-u[0] + 3.0 * u[1] - 3.0 * u[2] + u[3]) / smr3
            du[1] = (
                sigma
                * (3.0 * u[0] - 10.0 * u[1] + 12.0 * u[2] - 6.0 * u[3] + u[4])
                / smr2
            )
            du[2] = (
                sigma
                * (
                    -3.0 * u[0]
                    + 12.0 * u[1]
                    - 19.0 * u[2]
                    + 15.0 * u[3]
                    - 6.0 * u[4]
                    + u[5]
                )
                / smr1
            )
            du[-3] = (
                sigma
                * (
                    u[-6]
                    - 6.0 * u[-5]
                    + 15.0 * u[-4]
                    - 19.0 * u[-3]
                    + 12.0 * u[-2]
                    - 3.0 * u[-1]
                )
                / spr1
            )
            du[-2] = (
                sigma
                * (u[-5] - 6.0 * u[-4] + 12.0 * u[-3] - 10.0 * u[-2] + 3.0 * u[-1])
                / spr2
            )
            du[-1] = sigma * (u[-4] - 3.0 * u[-3] + 3.0 * u[-2] - u[-1]) / spr3


class KreissOligerFilterO8_1D(Filter1D):
    """
    Kreiss-Oliger filter in 1D
    """

    def __init__(self, dx, sigma, apply_diss_boundaries=True):
        self.sigma = sigma
        self.APPLY_DISS_BOUNDARIES = apply_diss_boundaries
        apply_filter = FilterApply.RHS
        filter_type = FilterType.KREISS_OLIGER_O8
        super().__init__(dx, apply_filter, filter_type)

    def get_sigma(self):
        return self.sigma

    def filter(self, du, u):
        # Kreiss-Oliger filter in x direction
        dx = self.dx
        sigma = self.sigma
        factor = -sigma / (256.0 * dx)

        # centered stencil
        du[4:-4] = factor * (
            u[:-8]
            - 8.0 * u[1:-7]
            + 28.0 * u[2:-6]
            - 56.0 * u[3:-5]
            + 70.0 * u[4:-4]
            - 56.0 * u[5:-3]
            + 28.0 * u[6:-2]
            - 8.0 * u[7:-1]
            + u[8:]
        )

        if self.APPLY_DISS_BOUNDARIES:
            smr4 = 17.0 / 48.0 * 256 * dx
            smr3 = 59.0 / 48.0 * 256 * dx
            smr2 = 43.0 / 48.0 * 256 * dx
            smr1 = 49.0 / 48.0 * 256 * dx
            spr4 = smr4
            spr3 = smr3
            spr2 = smr2
            spr1 = smr1
            du[0] = sigma * (-u[5] + 4.0 * u[4] - 6.0 * u[3] + 4.0 * u[1] - u[5]) / smr4
            du[1] = (
                sigma
                * (2.0 * u[4] - 9.0 * u[3] + 15.0 * u[2] - 11.0 * u[1] + 3.0 * u[0])
                / smr3
            )
            du[2] = (
                sigma
                * (-u[5] + 3.0 * u[4] - 8.0 * u[2] + 9.0 * u[1] - 3.0 * u[0])
                / smr2
            )
            du[3] = (
                sigma
                * (
                    -u[6]
                    + 6.0 * u[5]
                    - 14.0 * u[4]
                    + 15.0 * u[3]
                    - 6.0 * u[2]
                    - u[1]
                    + u[0]
                )
                / smr1
            )
            du[-1] = (
                sigma
                * (-u[-5] + 4.0 * u[-4] - 6.0 * u[-3] + 4.0 * u[-2] - u[-1])
                / spr4
            )
            du[-2] = (
                sigma
                * (2.0 * u[-5] - 9 * u[-4] + 15.0 * u[-3] - 11.0 * u[-2] + 3.0 * u[-1])
                / spr3
            )
            du[-3] = (
                sigma
                * (-u[-6] + 3.0 * u[-5] - 8.0 * u[-3] + 9.0 * u[-2] - 3.0 * u[-1])
                / spr2
            )
            du[-4] = (
                sigma
                * (
                    -u[-7]
                    + 6.0 * u[-6]
                    - 14.0 * u[-5]
                    + 15.0 * u[-4]
                    - 6.0 * u[-3]
                    - u[-2]
                    + u[-1]
                )
                / spr1
            )


"""
class CompactFilter2D(Filter2D):
    def __init__(self, x, y, type, use_banded=False):
        self.Nx = len(x)
        self.Ny = len(y)
        self.x = x
        self.y = y
        self.apply_filter = apply_filter
        self.filter_type = filter_type
        self.lusolve = use_banded
        self.F_x = CompactFilter(x, type, lusolve=use_banded)
        self.F_y = CompactFilter(y, type, lusolve=use_banded)
        super().__init__(dx, dy)

    def filter(self, u):
        filter_x_(u)
        filter_y_(u)

    def filter_x_(self, u):
        # Apply compact scheme row-wise (derivative in x)
        return self.F_x.grad(u)

    def filter_y_(self, u):
        # Apply compact scheme column-wise (derivative in y)
        du = self.F_y.grad(np.transpose(u))
        return np.transpose(du)
"""
