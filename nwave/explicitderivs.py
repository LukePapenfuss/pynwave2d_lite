import numpy as np
from numba import njit
from .types import DerivType
from .finitederivs import (
    FirstDerivative1D,
    SecondDerivative1D,
    FirstDerivative2D,
    SecondDerivative2D,
)


# ========================================================================
#   ExplicitFirst44_1D and ExplicitSecond44_1D
#
#   Classes for 1st and 2nd derivatives with 4th order accuracy in
#   one dimension.
# =======================================================================
class ExplicitFirst44_1D(FirstDerivative1D):
    def __init__(self, dx):
        self.type = DerivType.D1_E44
        super().__init__(dx, aderiv=True)

    def get_type(self):
        return self.type

    def grad(self, u):
        du = np.zeros_like(u)
        idx_by_12 = 1.0 / (12 * self.dx)

        # center stencil
        du[2:-2] = (-u[4:] + 8 * u[3:-1] - 8 * u[1:-3] + u[0:-4]) * idx_by_12

        # 4th order boundary stencils
        du[0] = (-25 * u[0] + 48 * u[1] - 36 * u[2] + 16 * u[3] - 3 * u[4]) * idx_by_12
        du[1] = (-3 * u[0] - 10 * u[1] + 18 * u[2] - 6 * u[3] + u[4]) * idx_by_12
        du[-2] = (-u[-5] + 6 * u[-4] - 18 * u[-3] + 10 * u[-2] + 3 * u[-1]) * idx_by_12
        du[-1] = (
            3 * u[-5] - 16 * u[-4] + 36 * u[-3] - 48 * u[-2] + 25 * u[-1]
        ) * idx_by_12

        return du

    @staticmethod
    @njit
    def advec_grad(du, u, beta, dx):
        idx_by_12 = 1.0 / (12 * dx)

        du[0] = (
            -25.0 * u[0] + 48.0 * u[1] - 36.0 * u[2] + 16.0 * u[3] - 3.0 * u[4]
        ) * idx_by_12
        du[1] = (
            -3.0 * u[0] - 10.0 * u[1] + 18.0 * u[2] - 6.0 * u[3] + u[4]
        ) * idx_by_12
        if beta[2] >= 0.0:
            du[2] = (
                -3.0 * u[1] - 10.0 * u[2] + 18.0 * u[3] - 6.0 * u[4] + u[5]
            ) * idx_by_12
        else:
            du[2] = (u[0] - 8.0 * u[1] + 8.0 * u[3] - u[4]) * idx_by_12

        for i in range(3, len(u) - 3):
            if beta[i] >= 0.0:
                du[i] = (
                    -3.0 * u[i - 1]
                    - 10.0 * u[i]
                    + 18.0 * u[i + 1]
                    - 6.0 * u[i + 2]
                    + u[i + 3]
                ) * idx_by_12
            else:
                du[i] = (
                    3.0 * u[i + 1]
                    + 10.0 * u[i]
                    - 18.0 * u[i - 1]
                    + 6.0 * u[i - 2]
                    - u[i - 3]
                ) * idx_by_12

        if beta[-3] >= 0.0:
            du[-3] = (u[-5] - 8.0 * u[-4] + 8.0 * u[-2] - u[-1]) * idx_by_12
        else:
            du[-3] = (
                3.0 * u[-2] + 10.0 * u[-3] - 18.0 * u[-4] + 6.0 * u[-5] - u[-6]
            ) * idx_by_12

        du[-2] = (
            -u[-5] + 6.0 * u[-4] - 18.0 * u[-3] + 10.0 * u[-2] + 3.0 * u[-1]
        ) * idx_by_12
        du[-1] = (
            3.0 * u[-5] - 16.0 * u[-4] + 36.0 * u[-3] - 48.0 * u[-2] + 25.0 * u[-1]
        ) * idx_by_12


class ExplicitSecond44_1D(SecondDerivative1D):
    def __init__(self, dx):
        self.type = DerivType.D2_E44
        super().__init__(dx)

    def get_type(self):
        return self.type

    def grad(self, u):
        idx_sqrd = 1.0 / self.dx**2
        idx_sqrd_by_12 = idx_sqrd / 12.0

        dxxu = np.zeros_like(u)
        dxxu[2:-2] = (
            -u[4:] + 16 * u[3:-1] - 30 * u[2:-2] + 16 * u[1:-3] - u[0:-4]
        ) * idx_sqrd_by_12

        # boundary stencils
        dxxu[0] = (
            45 * u[0] - 154 * u[1] + 214 * u[2] - 156 * u[3] + 61 * u[4] - 10 * u[5]
        ) * idx_sqrd_by_12
        dxxu[1] = (
            10 * u[0] - 15 * u[1] - 4 * u[2] + 14 * u[3] - 6 * u[4] + u[5]
        ) * idx_sqrd_by_12
        dxxu[-2] = (
            u[-6] - 6 * u[-5] + 14 * u[-4] - 4 * u[-3] - 15 * u[-2] + 10 * u[-1]
        ) * idx_sqrd_by_12
        dxxu[-1] = (
            -10 * u[-6]
            + 61 * u[-5]
            - 156 * u[-4]
            + 214 * u[-3]
            - 154 * u[-2]
            + 45 * u[-1]
        ) * idx_sqrd_by_12
        return dxxu


# ========================================================================
#   ExplicitFirst642_1D and ExplicitSecond642_1D
#
#   Classes for 1st and 2nd derivatives with 6th order accuracy in
#   one dimension.
# =======================================================================
class ExplicitFirst642_1D(FirstDerivative1D):
    def __init__(self, dx):
        self.type = DerivType.D1_E642
        super().__init__(dx, aderiv=True)

    def get_type(self):
        return self.type

    def grad(self, u):
        du = np.zeros_like(u)
        idx = 1.0 / self.dx
        idx_by_2 = 0.5 * idx
        idx_by_12 = idx / 12.0
        idx_by_60 = idx / 60.0

        du[0] = (-3.0 * u[0] + 4.0 * u[1] - u[2]) * idx_by_2
        du[1] = (-u[0] + u[2]) * idx_by_2
        du[2] = (u[0] - 8.0 * u[1] + 8.0 * u[3] - u[4]) * idx_by_12

        du[3:-3] = (
            -u[:-6]
            + 9.0 * u[1:-5]
            - 45.0 * u[2:-4]
            + 45.0 * u[4:-2]
            - 9.0 * u[5:-1]
            + u[6:]
        ) * idx_by_60

        du[-3] = (u[-5] - 8.0 * u[-4] + 8.0 * u[-2] - u[-1]) * idx_by_12
        du[-2] = (-u[-3] + u[-1]) * idx_by_2
        du[-1] = (u[-3] - 4.0 * u[-2] + 3.0 * u[-1]) * idx_by_2

        return du

    @staticmethod
    @njit
    def advec_grad(du, u, beta, dx):
        idx = 1.0 / dx
        idx_by_2 = 0.5 * idx
        idx_by_12 = idx / 12.0
        idx_by_60 = idx / 60.0

        du[0] = (-3.0 * u[0] + 4.0 * u[1] - u[2]) * idx_by_2

        if beta[1] >= 0.0:
            du[1] = (-3.0 * u[1] + 4.0 * u[2] - u[3]) * idx_by_2
        else:
            du[1] = (-u[0] + u[2]) * idx_by_2

        if beta[2] >= 0.0:
            du[2] = (
                -3.0 * u[1] - 10.0 * u[2] + 18.0 * u[3] - 6.0 * u[4] + u[5]
            ) * idx_by_12
        else:
            du[2] = (u[0] - 4.0 * u[1] + 3.0 * u[2]) * idx_by_2

        if beta[3] >= 0.0:
            du[3] = (
                2.0 * u[1]
                - 24.0 * u[2]
                - 35.0 * u[3]
                + 80.0 * u[4]
                - 30.0 * u[5]
                + 8.0 * u[6]
                - u[7]
            ) * idx_by_60
        else:
            du[3] = (
                -u[0] + 6.0 * u[1] - 18.0 * u[2] + 10.0 * u[3] + 3.0 * u[4]
            ) * idx_by_12

        # see gr-qc/0505055v2.pdf
        for i in range(4, len(u) - 4):
            if beta[i] >= 0.0:
                du[i] = (
                    2.0 * u[i - 2]
                    - 24.0 * u[i - 1]
                    - 35.0 * u[i]
                    + 80.0 * u[i + 1]
                    - 30.0 * u[i + 2]
                    + 8.0 * u[i + 3]
                    - u[i + 4]
                ) * idx_by_60
            else:
                du[i] = (
                    u[i - 4]
                    - 8.0 * u[i - 3]
                    + 30.0 * u[i - 2]
                    - 80.0 * u[i - 1]
                    + 35.0 * u[i]
                    + 24.0 * u[i + 1]
                    - 2.0 * u[i + 2]
                ) * idx_by_60

        if beta[-4] >= 0.0:
            du[-4] = (
                -3.0 * u[-5] - 10.0 * u[-4] + 18.0 * u[-3] - 6.0 * u[-2] + u[-1]
            ) * idx_by_12
        else:
            du[-4] = (
                u[-8]
                - 8.0 * u[-7]
                + 30.0 * u[-6]
                - 80.0 * u[-5]
                + 35.0 * u[-4]
                + 24.0 * u[-3]
                - 2.0 * u[-2]
            ) * idx_by_60

        if beta[-3] >= 0.0:
            du[-3] = (-3.0 * u[-3] + 4.0 * u[-2] - u[-1]) * idx_by_2
        else:
            du[-3] = (
                -u[-6] + 6.0 * u[-5] - 18.0 * u[-4] + 10.0 * u[-3] + 3.0 * u[-2]
            ) * idx_by_12

        if beta[-2] >= 0.0:
            du[-2] = (-u[-3] + u[-1]) * idx_by_2
        else:
            du[-2] = (u[-4] - 4.0 * u[-3] + 3.0 * u[-2]) * idx_by_2

        du[-1] = (u[-3] - 4.0 * u[-2] + 3.0 * u[-1]) * idx_by_2


class ExplicitSecond642_1D(SecondDerivative1D):
    def __init__(self, dx):
        self.type = DerivType.D1_E642
        super().__init__(dx)

    def get_type(self):
        return self.type

    def grad(self, u):
        dxxu = np.zeros_like(u)

        idx_sqrd = 1.00 / (self.dx * self.dx)
        idx_sqrd_by_12 = idx_sqrd / 12.0
        idx_sqrd_by_180 = idx_sqrd / 180.0

        dxxu[0] = (2.0 * u[0] - 5.0 * u[1] + 4.0 * u[2] - u[3]) * idx_sqrd
        dxxu[1] = (u[0] - 2.0 * u[1] + u[2]) * idx_sqrd
        dxxu[2] = (
            -u[0] + 16.0 * u[1] - 30.0 * u[2] + 16.0 * u[3] - u[4]
        ) * idx_sqrd_by_12

        dxxu[3:-3] = (
            2.0 * u[:-6]
            - 27.0 * u[1:-5]
            + 270.0 * u[2:-4]
            - 490.0 * u[3:-3]
            + 270.0 * u[4:-2]
            - 27.0 * u[5:-1]
            + 2.0 * u[6:]
        ) * idx_sqrd_by_180

        dxxu[-3] = (
            -u[-5] + 16.0 * u[-4] - 30.0 * u[-3] + 16.0 * u[-2] - u[-1]
        ) * idx_sqrd_by_12
        dxxu[-2] = (u[-3] - 2.0 * u[-2] + u[-1]) * idx_sqrd
        dxxu[-1] = (-u[-4] + 4.0 * u[-3] - 5.0 * u[-2] + 2.0 * u[-1]) * idx_sqrd

        return dxxu


# ========================================================================
#   ExplicitFirst42_2D and ExplicitSecond42_2D
#
#   Classes for 1st and 2nd derivatives with 4th order accuracy in
#   two dimensions.
# =======================================================================
class ExplicitFirst42_2D(FirstDerivative2D):
    def __init__(self, dx, dy):
        self.type = DerivType.D1_E42
        super().__init__(dx, dy)

    def grad_x(self, u):
        dudx = np.zeros_like(u)
        idx_by_2 = 1.0 / (2 * self.dx)
        idx_by_12 = 1.0 / (12 * self.dx)

        # center stencil
        dudx[2:-2, :] = (
            -u[4:, :] + 8 * u[3:-1, :] - 8 * u[1:-3, :] + u[0:-4, :]
        ) * idx_by_12

        # 4th order boundary stencils
        dudx[0, :] = (-3 * u[0, :] + 4 * u[1, :] - u[2, :]) * idx_by_2
        dudx[1, :] = (-u[0, :] + u[2, :]) * idx_by_2
        dudx[-2, :] = (-u[-3, :] + u[-1, :]) * idx_by_2
        dudx[-1, :] = (u[-3, :] - 4 * u[-2, :] + 3 * u[-1, :]) * idx_by_2

        return dudx

    def grad_y(self, u) -> np.ndarray:
        dudy = np.zeros_like(u)
        idy_by_2 = 1.0 / (2 * self.dy)
        idy_by_12 = 1.0 / (12 * self.dy)

        # center stencil
        dudy[:, 2:-2] = (
            -u[:, 4:] + 8 * u[:, 3:-1] - 8 * u[:, 1:-3] + u[:, 0:-4]
        ) * idy_by_12

        # 4th order boundary stencils
        dudy[:, 0] = (-3 * u[:, 0] + 4 * u[:, 1] - u[:, 2]) * idy_by_2
        dudy[:, 1] = (-u[:, 0] + u[:, 2]) * idy_by_2
        dudy[:, -2] = (-u[:, -3] + u[:, -1]) * idy_by_2
        dudy[:, -1] = (u[:, -3] - 4 * u[:, -2] + 3 * u[:, -1]) * idy_by_2

        return dudy


class ExplicitSecond42_2D(SecondDerivative2D):
    def __init__(self, dx, dy):
        self.type = DerivType.D2_E42
        super().__init__(dx, dy)

    def grad_xx(self, u):
        idx_sqrd = 1.0 / self.dx**2
        idx_sqrd_by_12 = idx_sqrd / 12.0

        dxxu = np.zeros_like(u)
        dxxu[2:-2, :] = (
            -u[4:, :] + 16 * u[3:-1, :] - 30 * u[2:-2, :] + 16 * u[1:-3, :] - u[0:-4, :]
        ) * idx_sqrd_by_12

        # boundary stencils
        dxxu[0, :] = (2 * u[0, :] - 5 * u[1, :] + 4 * u[2, :] - u[3, :]) * idx_sqrd
        dxxu[1, :] = (u[0, :] - 2 * u[1, :] + u[2, :]) * idx_sqrd
        dxxu[-2, :] = (u[-3, :] - 2 * u[-2, :] + u[-1, :]) * idx_sqrd
        dxxu[-1, :] = (
            -u[-4, :] + 4 * u[-3, :] - 5 * u[-2, :] + 2 * u[-1, :]
        ) * idx_sqrd
        return dxxu

    def grad_yy(self, u):
        idy_sqrd = 1.0 / self.dy**2
        idy_sqrd_by_12 = idy_sqrd / 12.0
        dyyu = np.zeros_like(u)

        # centered stencils
        dyyu[:, 2:-2] = (
            -u[:, 4:] + 16 * u[:, 3:-1] - 30 * u[:, 2:-2] + 16 * u[:, 1:-3] - u[:, 0:-4]
        ) * idy_sqrd_by_12

        # boundary stencils
        dyyu[:, 0] = (2 * u[:, 0] - 5 * u[:, 1] + 4 * u[:, 2] - u[:, 3]) * idy_sqrd
        dyyu[:, 1] = (u[:, 0] - 2 * u[:, 1] + u[:, 2]) * idy_sqrd
        dyyu[:, -2] = (u[:, -3] - 2 * u[:, -2] + u[:, -1]) * idy_sqrd
        dyyu[:, -1] = (
            -u[:, -4] + 4 * u[:, -3] - 5 * u[:, -2] + 2 * u[:, -1]
        ) * idy_sqrd
        return dyyu


# ========================================================================
#   ExplicitFirst44_2D and ExplicitSecond44_2D
#
#   Classes for 1st and 2nd derivatives with 4th order accuracy in
#   two dimensions.
# =======================================================================
class ExplicitFirst44_2D(FirstDerivative2D):
    def __init__(self, dx, dy):
        self.type = DerivType.D1_E44
        super().__init__(dx, dy)

    def grad_x(self, u):
        dudx = np.zeros_like(u)
        idx_by_12 = 1.0 / (12 * self.dx)

        # center stencil
        dudx[2:-2, :] = (
            -u[4:, :] + 8 * u[3:-1, :] - 8 * u[1:-3, :] + u[0:-4, :]
        ) * idx_by_12

        # 4th order boundary stencils
        dudx[0, :] = (
            -25 * u[0, :] + 48 * u[1, :] - 36 * u[2, :] + 16 * u[3, :] - 3 * u[4, :]
        ) * idx_by_12
        dudx[1, :] = (
            -3 * u[0, :] - 10 * u[1, :] + 18 * u[2, :] - 6 * u[3, :] + u[4, :]
        ) * idx_by_12
        dudx[-2, :] = (
            -u[-5, :] + 6 * u[-4, :] - 18 * u[-3, :] + 10 * u[-2, :] + 3 * u[-1, :]
        ) * idx_by_12
        dudx[-1, :] = (
            3 * u[-5, :] - 16 * u[-4, :] + 36 * u[-3, :] - 48 * u[-2, :] + 25 * u[-1, :]
        ) * idx_by_12

        return dudx

    def grad_y(self, u) -> np.ndarray:
        dudy = np.zeros_like(u)
        idy_by_12 = 1.0 / (12 * self.dy)

        # center stencil
        dudy[:, 2:-2] = (
            -u[:, 4:] + 8 * u[:, 3:-1] - 8 * u[:, 1:-3] + u[:, 0:-4]
        ) * idy_by_12

        # 4th order boundary stencils
        dudy[:, 0] = (
            -25 * u[:, 0] + 48 * u[:, 1] - 36 * u[:, 2] + 16 * u[:, 3] - 3 * u[:, 4]
        ) * idy_by_12
        dudy[:, 1] = (
            -3 * u[:, 0] - 10 * u[:, 1] + 18 * u[:, 2] - 6 * u[:, 3] + u[:, 4]
        ) * idy_by_12
        dudy[:, -2] = (
            -u[:, -5] + 6 * u[:, -4] - 18 * u[:, -3] + 10 * u[:, -2] + 3 * u[:, -1]
        ) * idy_by_12
        dudy[:, -1] = (
            3 * u[:, -5] - 16 * u[:, -4] + 36 * u[:, -3] - 48 * u[:, -2] + 25 * u[:, -1]
        ) * idy_by_12

        return dudy


class ExplicitSecond44_2D(SecondDerivative2D):
    def __init__(self, dx, dy):
        self.type = DerivType.D2_E44
        super().__init__(dx, dy)

    def grad_xx(self, u):
        idx_sqrd = 1.0 / self.dx**2
        idx_sqrd_by_12 = idx_sqrd / 12.0

        dxxu = np.zeros_like(u)
        dxxu[2:-2, :] = (
            -u[4:, :] + 16 * u[3:-1, :] - 30 * u[2:-2, :] + 16 * u[1:-3, :] - u[0:-4, :]
        ) * idx_sqrd_by_12

        # boundary stencils
        dxxu[0, :] = (
            45 * u[0, :]
            - 154 * u[1, :]
            + 214 * u[2, :]
            - 156 * u[3, :]
            + 61 * u[4, :]
            - 10 * u[5, :]
        ) * idx_sqrd_by_12
        dxxu[1, :] = (
            10 * u[0, :]
            - 15 * u[1, :]
            - 4 * u[2, :]
            + 14 * u[3, :]
            - 6 * u[4, :]
            + u[5, :]
        ) * idx_sqrd_by_12
        dxxu[-2, :] = (
            u[-6, :]
            - 6 * u[-5, :]
            + 14 * u[-4, :]
            - 4 * u[-3, :]
            - 15 * u[-2, :]
            + 10 * u[-1, :]
        ) * idx_sqrd_by_12
        dxxu[-1, :] = (
            -10 * u[-6, :]
            + 61 * u[-5, :]
            - 156 * u[-4, :]
            + 214 * u[-3, :]
            - 154 * u[-2, :]
            + 45 * u[-1, :]
        ) * idx_sqrd_by_12
        return dxxu

    def grad_yy(self, u):
        idy_sqrd = 1.0 / self.dy**2
        idy_sqrd_by_12 = idy_sqrd / 12.0
        dyyu = np.zeros_like(u)

        # centered stencils
        dyyu[:, 2:-2] = (
            -u[:, 4:] + 16 * u[:, 3:-1] - 30 * u[:, 2:-2] + 16 * u[:, 1:-3] - u[:, 0:-4]
        ) * idy_sqrd_by_12

        # boundary stencils
        dyyu[:, 0] = (
            45 * u[:, 0]
            - 154 * u[:, 1]
            + 214 * u[:, 2]
            - 156 * u[:, 3]
            + 61 * u[:, 4]
            - 10 * u[:, 5]
        ) * idy_sqrd_by_12
        dyyu[:, 1] = (
            10 * u[:, 0]
            - 15 * u[:, 1]
            - 4 * u[:, 2]
            + 14 * u[:, 3]
            - 6 * u[:, 4]
            + u[:, 5]
        ) * idy_sqrd_by_12
        dyyu[:, -2] = (
            u[:, -6]
            - 6 * u[:, -5]
            + 14 * u[:, -4]
            - 4 * u[:, -3]
            - 15 * u[:, -2]
            + 10 * u[:, -1]
        ) * idy_sqrd_by_12
        dyyu[:, -1] = (
            -10 * u[:, -6]
            + 61 * u[:, -5]
            - 156 * u[:, -4]
            + 214 * u[:, -3]
            - 154 * u[:, -2]
            + 45 * u[:, -1]
        ) * idy_sqrd_by_12
        return dyyu


# ========================================================================
#   ExplicitFirst642_2D and ExplicitSecond642_2D
#
#   Classes for 1st and 2nd derivatives with 6th order accuracy in
#   two dimensions.
# =======================================================================
class ExplicitFirst642_2D(FirstDerivative2D):
    def __init__(self, dx, dy):
        self.type = DerivType.D2_E642
        super().__init__(dx, dy)

    def get_type(self):
        return self.type

    def grad_x(self, u):
        du = np.zeros_like(u)
        idx = 1.0 / self.dx
        idx_by_2 = 0.5 * idx
        idx_by_12 = idx / 12.0
        idx_by_60 = idx / 60.0

        du[0, :] = (-3.0 * u[0, :] + 4.0 * u[1, :] - u[2]) * idx_by_2
        du[1, :] = (-u[0, :] + u[2, :]) * idx_by_2
        du[2, :] = (u[0, :] - 8.0 * u[1, :] + 8.0 * u[3, :] - u[4, :]) * idx_by_12

        du[3:-3, :] = (
            -u[:-6, :]
            + 9.0 * u[1:-5, :]
            - 45.0 * u[2:-4, :]
            + 45.0 * u[4:-2, :]
            - 9.0 * u[5:-1, :]
            + u[6:, :]
        ) * idx_by_60

        du[-3, :] = (u[-5, :] - 8.0 * u[-4, :] + 8.0 * u[-2, :] - u[-1, :]) * idx_by_12
        du[-2, :] = (-u[-3, :] + u[-1, :]) * idx_by_2
        du[-1, :] = (u[-3, :] - 4.0 * u[-2, :] + 3.0 * u[-1, :]) * idx_by_2

        return du

    def grad_y(self, u):
        du = np.zeros_like(u)
        idy = 1.0 / self.dy
        idy_by_2 = 0.5 * idy
        idy_by_12 = idy / 12.0
        idy_by_60 = idy / 60.0

        du[:, 0] = (-3.0 * u[:, 0] + 4.0 * u[:, 1] - u[:, 2]) * idy_by_2
        du[:, 1] = (-u[:, 0] + u[:, 2]) * idy_by_2
        du[:, 2] = (u[:, 0] - 8.0 * u[:, 1] + 8.0 * u[:, 3] - u[:, 4]) * idy_by_12

        du[:, 3:-3] = (
            -u[:, :-6]
            + 9.0 * u[:, 1:-5]
            - 45.0 * u[:, 2:-4]
            + 45.0 * u[:, 4:-2]
            - 9.0 * u[:, 5:-1]
            + u[:, 6:]
        ) * idy_by_60

        du[:, -3] = (u[:, -5] - 8.0 * u[:, -4] + 8.0 * u[:, -2] - u[:, -1]) * idy_by_12
        du[:, -2] = (-u[:, -3] + u[:, -1]) * idy_by_2
        du[:, -1] = (u[:, -3] - 4.0 * u[:, -2] + 3.0 * u[:, -1]) * idy_by_2

        return du


class ExplicitSecond642_2D(SecondDerivative2D):
    def __init__(self, dx, dy):
        self.type = DerivType.D2_E642
        super().__init__(dx, dy)

    def get_type(self):
        return self.type

    def grad_xx(self, u):
        dxxu = np.zeros_like(u)

        idx_sqrd = 1.00 / (self.dx * self.dx)
        idx_sqrd_by_12 = idx_sqrd / 12.0
        idx_sqrd_by_180 = idx_sqrd / 180.0

        dxxu[0, :] = (
            2.0 * u[0, :] - 5.0 * u[1, :] + 4.0 * u[2, :] - u[3, :]
        ) * idx_sqrd
        dxxu[1, :] = (u[0, :] - 2.0 * u[1, :] + u[2, :]) * idx_sqrd
        dxxu[2, :] = (
            -u[0, :] + 16.0 * u[1, :] - 30.0 * u[2, :] + 16.0 * u[3, :] - u[4, :]
        ) * idx_sqrd_by_12

        dxxu[3:-3, :] = (
            2.0 * u[:-6, :]
            - 27.0 * u[1:-5, :]
            + 270.0 * u[2:-4, :]
            - 490.0 * u[3:-3, :]
            + 270.0 * u[4:-2, :]
            - 27.0 * u[5:-1, :]
            + 2.0 * u[6:, :]
        ) * idx_sqrd_by_180

        dxxu[-3, :] = (
            -u[-5, :] + 16.0 * u[-4, :] - 30.0 * u[-3, :] + 16.0 * u[-2, :] - u[-1, :]
        ) * idx_sqrd_by_12
        dxxu[-2, :] = (u[-3, :] - 2.0 * u[-2, :] + u[-1, :]) * idx_sqrd
        dxxu[-1, :] = (
            -u[-4, :] + 4.0 * u[-3, :] - 5.0 * u[-2, :] + 2.0 * u[-1, :]
        ) * idx_sqrd

        return dxxu

    def grad_yy(self, u):
        dyyu = np.zeros_like(u)

        idy_sqrd = 1.00 / (self.dy * self.dy)
        idy_sqrd_by_12 = idy_sqrd / 12.0
        idy_sqrd_by_180 = idy_sqrd / 180.0

        dyyu[:, 0] = (
            2.0 * u[:, 0] - 5.0 * u[:, 1] + 4.0 * u[:, 2] - u[:, 3]
        ) * idy_sqrd
        dyyu[:, 1] = (u[:, 0] - 2.0 * u[:, 1] + u[:, 2]) * idy_sqrd
        dyyu[:, 2] = (
            -u[:, 0] + 16.0 * u[:, 1] - 30.0 * u[:, 2] + 16.0 * u[:, 3] - u[:, 4]
        ) * idy_sqrd_by_12

        dyyu[:, 3:-3] = (
            2.0 * u[:, :-6]
            - 27.0 * u[:, 1:-5]
            + 270.0 * u[:, 2:-4]
            - 490.0 * u[:, 3:-3]
            + 270.0 * u[:, 4:-2]
            - 27.0 * u[:, 5:-1]
            + 2.0 * u[:, 6:]
        ) * idy_sqrd_by_180

        dyyu[:, -3] = (
            -u[:, -5] + 16.0 * u[:, -4] - 30.0 * u[:, -3] + 16.0 * u[:, -2] - u[:, -1]
        ) * idy_sqrd_by_12
        dyyu[:, -2] = (u[:, -3] - 2.0 * u[:, -2] + u[:, -1]) * idy_sqrd
        dyyu[:, -1] = (
            -u[:, -4] + 4.0 * u[:, -3] - 5.0 * u[:, -2] + 2.0 * u[:, -1]
        ) * idy_sqrd

        return dyyu


# ========================================================================
#   ExplicitFirst666_2D and ExplicitSecond666_2D
#
#   Classes for 1st and 2nd derivatives with 6th order accuracy in
#   two dimensions.
# =======================================================================
class ExplicitFirst666_2D(FirstDerivative2D):
    def __init__(self, dx, dy):
        self.type = DerivType.D1_E666
        super().__init__(dx, dy)

    def get_type(self):
        return self.type

    def grad_x(self, u):
        du = np.zeros_like(u)
        idx = 1.0 / self.dx
        idx_by_60 = idx / 60.0

        du[0, :] = (
            -147.0 * u[0, :]
            + 360.0 * u[1, :]
            - 450.0 * u[2, :]
            + 400.0 * u[3, :]
            - 225.0 * u[4, :]
            + 72.0 * u[5, :]
            - 10.0 * u[6, :]
        ) * idx_by_60
        du[1, :] = (
            -10.0 * u[0, :]
            - 77.0 * u[1, :]
            + 150.0 * u[2, :]
            - 100.0 * u[3, :]
            + 50.0 * u[4, :]
            - 15.0 * u[5, :]
            + 2.0 * u[6, :]
        ) * idx_by_60
        du[2, :] = (
            2.0 * u[0, :]
            - 24.0 * u[1, :]
            - 35.0 * u[2, :]
            + 80.0 * u[3, :]
            - 30.0 * u[4, :]
            + 8.0 * u[5, :]
            - u[6, :]
        ) * idx_by_60

        du[3:-3, :] = (
            -u[:-6, :]
            + 9.0 * u[1:-5, :]
            - 45.0 * u[2:-4, :]
            + 45.0 * u[4:-2, :]
            - 9.0 * u[5:-1, :]
            + u[6:, :]
        ) * idx_by_60

        du[-3, :] = (
            u[-7, :]
            - 8.0 * u[-6, :]
            + 30.0 * u[-5, :]
            - 80.0 * u[-4, :]
            + 35.0 * u[-3, :]
            + 24.0 * u[-2, :]
            - 2.0 * u[-1, :]
        ) * idx_by_60
        du[-2, :] = (
            -2.0 * u[-7, :]
            + 15.0 * u[-6, :]
            - 50.0 * u[-5, :]
            + 100.0 * u[-4, :]
            - 150.0 * u[-3, :]
            + 77.0 * u[-2, :]
            + 10.0 * u[-1, :]
        ) * idx_by_60
        du[-1, :] = (
            10.0 * u[-7, :]
            - 72.0 * u[-6, :]
            + 225.0 * u[-5, :]
            - 400.0 * u[-4, :]
            + 450.0 * u[-3, :]
            - 360.0 * u[-2, :]
            + 147.0 * u[-1, :]
        ) * idx_by_60

        return du

    def grad_y(self, u):
        du = np.zeros_like(u)
        idy = 1.0 / self.dy
        idy_by_60 = idy / 60.0

        du[:, 0] = (
            -147.0 * u[:, 0]
            + 360.0 * u[:, 1]
            - 450.0 * u[:, 2]
            + 400.0 * u[:, 3]
            - 225.0 * u[:, 4]
            + 72.0 * u[:, 5]
            - 10.0 * u[:, 6]
        ) * idy_by_60
        du[:, 1] = (
            -10.0 * u[:, 0]
            - 77.0 * u[:, 1]
            + 150.0 * u[:, 2]
            - 100.0 * u[:, 3]
            + 50.0 * u[:, 4]
            - 15.0 * u[:, 5]
            + 2.0 * u[:, 6]
        ) * idy_by_60
        du[:, 2] = (
            2.0 * u[:, 0]
            - 24.0 * u[:, 1]
            - 35.0 * u[:, 2]
            + 80.0 * u[:, 3]
            - 30.0 * u[:, 4]
            + 8.0 * u[:, 5]
            - u[:, 6]
        ) * idy_by_60

        du[:, 3:-3] = (
            -u[:, :-6]
            + 9.0 * u[:, 1:-5]
            - 45.0 * u[:, 2:-4]
            + 45.0 * u[:, 4:-2]
            - 9.0 * u[:, 5:-1]
            + u[:, 6:]
        ) * idy_by_60

        du[:, -3] = (
            u[:, -7]
            - 8.0 * u[:, -6]
            + 30.0 * u[:, -5]
            - 80.0 * u[:, -4]
            + 35.0 * u[:, -3]
            + 24.0 * u[:, -2]
            - 2.0 * u[:, -1]
        ) * idy_by_60
        du[:, -2] = (
            -2.0 * u[:, -7]
            + 15.0 * u[:, -6]
            - 50.0 * u[:, -5]
            + 100.0 * u[:, -4]
            - 150.0 * u[:, -3]
            + 77.0 * u[:, -2]
            + 10.0 * u[:, -1]
        ) * idy_by_60
        du[:, -1] = (
            10.0 * u[:, -7]
            - 72.0 * u[:, -6]
            + 225.0 * u[:, -5]
            - 400.0 * u[:, -4]
            + 450.0 * u[:, -3]
            - 360.0 * u[:, -2]
            + 147.0 * u[:, -1]
        ) * idy_by_60

        return du


class ExplicitSecond666_2D(SecondDerivative2D):
    def __init__(self, dx, dy):
        self.type = DerivType.D2_E666
        super().__init__(dx, dy)

    def get_type(self):
        return self.type

    def grad_xx(self, u):
        dxxu = np.zeros_like(u)

        idx_sqrd = 1.00 / (self.dx * self.dx)
        idx_sqrd_by_180 = idx_sqrd / 180.0

        dxxu[0, :] = (
            938.0 * u[0, :]
            - 4014.0 * u[1, :]
            + 7911.0 * u[2, :]
            - 9490.0 * u[3, :]
            + 7380.0 * u[4, :]
            - 3618.0 * u[5, :]
            + 1019.0 * u[6, :]
            - 126.0 * u[7, :]
        ) * idx_sqrd_by_180
        dxxu[1, :] = (
            126.0 * u[0, :]
            - 70.0 * u[1, :]
            - 486.0 * u[2, :]
            + 855.0 * u[3, :]
            - 670.0 * u[4, :]
            + 324.0 * u[5, :]
            - 90.0 * u[6, :]
            + 11.0 * u[7, :]
        ) * idx_sqrd_by_180
        dxxu[2, :] = (
            -11.0 * u[0, :]
            + 214.0 * u[1, :]
            - 378.0 * u[2, :]
            + 130.0 * u[3, :]
            + 85.0 * u[4, :]
            - 54.0 * u[5, :]
            + 16.0 * u[6, :]
            - 2.0 * u[7, :]
        ) * idx_sqrd_by_180

        dxxu[3:-3, :] = (
            2.0 * u[:-6, :]
            - 27.0 * u[1:-5, :]
            + 270.0 * u[2:-4, :]
            - 490.0 * u[3:-3, :]
            + 270.0 * u[4:-2, :]
            - 27.0 * u[5:-1, :]
            + 2.0 * u[6:, :]
        ) * idx_sqrd_by_180

        dxxu[-3, :] = (
            -2.0 * u[-8, :]
            + 16.0 * u[-7, :]
            - 54.0 * u[-6, :]
            + 85.0 * u[-5, :]
            + 130.0 * u[-4, :]
            - 378.0 * u[-3, :]
            + 214.0 * u[-2, :]
            - 11.0 * u[-1, :]
        ) * idx_sqrd_by_180
        dxxu[-2, :] = (
            11.0 * u[-8, :]
            - 90.0 * u[-7, :]
            + 324.0 * u[-6, :]
            - 670.0 * u[-5, :]
            + 855.0 * u[-4, :]
            - 486.0 * u[-3, :]
            - 70.0 * u[-2, :]
            + 126.0 * u[-1, :]
        ) * idx_sqrd_by_180
        dxxu[-1, :] = (
            -126.0 * u[-8, :]
            + 1019.0 * u[-7, :]
            - 3618.0 * u[-6, :]
            + 7380.0 * u[-5, :]
            - 9490.0 * u[-4, :]
            + 7911.0 * u[-3, :]
            - 4014.0 * u[-2, :]
            + 938.0 * u[-1, :]
        ) * idx_sqrd_by_180

        return dxxu

    def grad_yy(self, u):
        dyyu = np.zeros_like(u)
        idy_sqrd = 1.00 / (self.dy * self.dy)
        idy_sqrd_by_180 = idy_sqrd / 180.0

        dyyu[:, 0] = (
            938.0 * u[:, 0]
            - 4014.0 * u[:, 1]
            + 7911.0 * u[:, 2]
            - 9490.0 * u[:, 3]
            + 7380.0 * u[:, 4]
            - 3618.0 * u[:, 5]
            + 1019.0 * u[:, 6]
            - 126.0 * u[:, 7]
        ) * idy_sqrd_by_180
        dyyu[:, 1] = (
            126.0 * u[:, 0]
            - 70.0 * u[:, 1]
            - 486.0 * u[:, 2]
            + 855.0 * u[:, 3]
            - 670.0 * u[:, 4]
            + 324.0 * u[:, 5]
            - 90.0 * u[:, 6]
            + 11.0 * u[:, 7]
        ) * idy_sqrd_by_180
        dyyu[:, 2] = (
            -11.0 * u[:, 0]
            + 214.0 * u[:, 1]
            - 378.0 * u[:, 2]
            + 130.0 * u[:, 3]
            + 85.0 * u[:, 4]
            - 54.0 * u[:, 5]
            + 16.0 * u[:, 6]
            - 2.0 * u[:, 7]
        ) * idy_sqrd_by_180

        dyyu[:, 3:-3] = (
            2.0 * u[:, :-6]
            - 27.0 * u[:, 1:-5]
            + 270.0 * u[:, 2:-4]
            - 490.0 * u[:, 3:-3]
            + 270.0 * u[:, 4:-2]
            - 27.0 * u[:, 5:-1]
            + 2.0 * u[:, 6:]
        ) * idy_sqrd_by_180

        dyyu[:, -3] = (
            -2.0 * u[:, -8]
            + 16.0 * u[:, -7]
            - 54.0 * u[:, -6]
            + 85.0 * u[:, -5]
            + 130.0 * u[:, -4]
            - 378.0 * u[:, -3]
            + 214.0 * u[:, -2]
            - 11.0 * u[:, -1]
        ) * idy_sqrd_by_180
        dyyu[:, -2] = (
            11.0 * u[:, -8]
            - 90.0 * u[:, -7]
            + 324.0 * u[:, -6]
            - 670.0 * u[:, -5]
            + 855.0 * u[:, -4]
            - 486.0 * u[:, -3]
            - 70.0 * u[:, -2]
            + 126.0 * u[:, -1]
        ) * idy_sqrd_by_180
        dyyu[:, -1] = (
            -126.0 * u[:, -8]
            + 1019.0 * u[:, -7]
            - 3618.0 * u[:, -6]
            + 7380.0 * u[:, -5]
            - 9490.0 * u[:, -4]
            + 7911.0 * u[:, -3]
            - 4014.0 * u[:, -2]
            + 938.0 * u[:, -1]
        ) * idy_sqrd_by_180

        return dyyu
