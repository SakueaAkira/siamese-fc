function run_tracker(video, visualization,gpus)
% RUN_TRACKER  is the external function of the tracker - does initialization and calls tracker.m
    startup;
    %% Parameters that should have no effect on the result.
    params.video = video;
    params.visualization = visualization;
    params.gpus = gpus;
    %% Parameters that should be recorded.
    % params.foo = 'blah';
    %% Call the main tracking function
    tracker(params);    
end