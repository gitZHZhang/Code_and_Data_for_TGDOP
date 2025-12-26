function P = generate_P(a,b,theta)

P0 = [a^2,0;0,b^2];
R = [cos(theta),-sin(theta);sin(theta),cos(theta)];
P = R*P0*R';