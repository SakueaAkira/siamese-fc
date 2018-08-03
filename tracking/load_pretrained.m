% -------------------------------------------------------------------------------------------------
function net = load_pretrained(netPath, gpu)
%LOAD_PRETRAINED loads a pretrained fully-convolutional Siamese network as a DagNN
% -------------------------------------------------------------------------------------------------
    % to keep consistency when reading all hyperparams
    if iscell(netPath)
        netPath = netPath{1};
    end
    trainResults = load(netPath);
    net = trainResults.net;
	
    % 如果过去版本的field存在则移除他们。
    [~,xcorrId] = find_layers_from_type(net, 'XCorr');
    xcorrId = xcorrId{1};
    if isfield(net.layers(xcorrId).block, 'expect')
        net.layers(xcorrId).block = rmfield(net.layers(xcorrId).block,'expect');
    end
    if isfield(net.layers(xcorrId).block, 'visualization_active')
        net.layers(xcorrId).block = rmfield(net.layers(xcorrId).block,'visualization_active');
    end
    if isfield(net.layers(xcorrId).block, 'visualization_grid_sz')
        net.layers(xcorrId).block = rmfield(net.layers(xcorrId).block,'visualization_grid_sz');
    end

    % 作为 dagNN 实例加载网络。
    net = dagnn.DagNN.loadobj(net);
    % remove loss layer
    net = remove_layers_from_block(net, 'dagnn.Loss');
    % init specified GPU
    if ~isempty(gpu)
       gpuDevice(gpu)
    end
	% 将网络数据移动到GPU中。
    net.move('gpu');
	% 选择为测试模式而不是训练模式。
    net.mode = 'test'; % very important for batch norm, we now use the stats accumulated during training.
end