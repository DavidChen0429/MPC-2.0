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
%InvSet.plot()

% Plot projections of the invariant set
figure;
InvSet.projection([1, 2]).plot(); % Plot projection onto the first two dimensions
title('Projection onto x1-x2 plane');

system = LTISystem('A', [1 1; 0 0.9], 'B', [1; 0.5]);
system.x.min = [-5; -5];
system.x.max = [5; 5];
system.u.min = -1;
system.u.max = 1;
InvSet = system.invariantSet()
InvSet.plot()
