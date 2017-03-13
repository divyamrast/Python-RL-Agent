function walker(file)
    addpath('/home/ivan/work/scripts/matlab/exporters/');
    Vref = 0.12;
    
    if nargin < 1
        file = 'compass_walker_*.csv';
    end
    D = readCsv(file);
    
    theta   = D(1:end, 2);
    dtheta  = D(1:end, 4);
    u       = D(1:end, end-2);
    contact = D(1:end, 6);
    hipVx   = - dtheta.*cos(theta);
        
%     disp(calcWalkerCost(0.2, Vref, theta, dtheta, u, 0.97, contact, 20));
%     disp(calcLC(hipVx, contact));
    
    windowSize = 200;
    b = (1/windowSize)*ones(1,windowSize);
    mvHipVx = filter(b,1,hipVx);
    
    disp(mean(hipVx(100:199)));
        
    figure(1);
    plot(D(1:end, 1), hipVx, 'b', D(1:end, 1), mvHipVx, 'r');
    grid on;
    xlabel('Timestep');
    ylabel('Hip velocity, m/s');
    title(sprintf('Simplest walker tracking Vref = %f', Vref));
    legend('Velocity', 'Moving average', 'Location', 'Best');%%

end