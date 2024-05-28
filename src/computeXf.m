% Define matrix A
A = [
    0   0   0   0   0   0   1   0   0   0   0   0;
    0   0   0   0   0   0   0   1   0   0   0   0;
    0   0   0   0   0   0   0   0   1   0   0   0;
    0   0   0   0   0   0   0   0   0   1   0   0;
    0   0   0   0   0   0   0   0   0   0   1   0;
    0   0   0   0   0   0   0   0   0   0   0   1;
    0   0   0   0  10   0   0   0   0   0   0   0;
    0   0   0  10   0   0   0   0   0   0   0   0;
    0   0   0  -0  -0   0   0   0   0   0   0   0;
    0   0   0   0   0   0   0   0   0   0   0   0;
    0   0   0   0   0   0   0   0   0  -0   0  -0;
    0   0   0   0   0   0   0   0   0   0   0   0;
];

% Define matrix B
B = [
    0           0           0           0;
    0           0           0           0;
    0           0           0           0;
    0           0           0           0;
    0           0           0           0;
    0           0           0           0;
    0           0           0           0;
    0           0           0           0;
    1           0           0           0;
    0    9.09090909    0           0;
    0           0    9.09090909    0;
    0           0           0   25;
];

mpt_init
system = LTISystem('A', A, 'B', B);
system.u.min = [-10; -1.4715; -1.4715; -0.0196];
system.u.max = [30; 1.4715; 1.4715; 0.0196];
system.x.max = [100; 100; 100; pi/2; pi/2; 2*pi; 2; 2; 2; 2*pi; 2*pi; 2*pi];
system.x.min = -1*[100; 100; 100; pi/2; pi/2; 2*pi; 2; 2; 2; 2*pi; 2*pi; 2*pi];
InvSet = system.invariantSet();

% Plot projections of the invariant set
figure;
InvSet.projection([1, 2]).plot(); % Plot projection onto the first two dimensions
title('Projection onto x1-x2 plane');

%% Simple 2D system
system = LTISystem('A', [1 1; 0 0.9], 'B', [1; 0.5]);
system.x.min = [-5; -5];
system.x.max = [5; 5];
system.u.min = -1;
system.u.max = 1;
InvSet = system.invariantSet();
InvSet.plot()

%% Debug
dim.nx = 16;        %
dim.nu = 4;         %
u_limit = 0.1;      % bound on control inputs
x_limit = 5;        % bound on states

Fu = kron(eye(dim.nu),[1; -1]); % 8,4
Fx = kron(eye(dim.nx),[1; -1]); % 32,16

fu = u_limit*ones(2*dim.nu,1);
fx = x_limit*ones(2*dim.nx,1);

f = [fu; fx];       % 40,1
F = blkdiag(Fu,Fx); % 40,20
s = size(F,1);      % 2*(nx+nu)