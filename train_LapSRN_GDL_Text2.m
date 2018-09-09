function train_LapSRN_GDL_Text(scale, depth,gdl_lambda, gpu)
% -------------------------------------------------------------------------
%   Description:
%       Script to train LapSRN from scratch
%
%   Input:
%       - scale : SR upsampling scale
%       - depth : numbers of conv layers in each pyramid level
%       - gpu   : GPU ID, 0 for CPU mode
%
%   
% Modify code from 
%       Wei-Sheng Lai
%       wlai24@ucmerced.edu
%       University of California, Merced
% -------------------------------------------------------------------------


    %% initialize opts
    opts = init_LapSRN_GDL_Text2_opts(scale, depth,gdl_lambda,gpu);

    %% save opts
    filename = fullfile(opts.train.expDir, 'opts.mat');
    fprintf('Save parameter %s\n', filename);
    save(filename, 'opts');

    %% setup paths
    addpath(genpath('utils'));
    addpath(fullfile(pwd, 'matconvnet/matlab'));
    vl_setupnn;

    %% initialize network
    fprintf('Initialize network...\n');
    model_filename = fullfile(opts.train.expDir, 'net-epoch-0.mat');
    if( ~exist(model_filename, 'file') )
        model = init_LapSRN_GDL_TextModel(opts);
        fprintf('Save %s\n', model_filename);
        net = model.saveobj();
        save(model_filename, 'net');
    else
        fprintf('Load %s\n', model_filename);
        model = load(model_filename);
        model = dagnn.DagNN.loadobj(model.net);
    end

    %% load imdb
    imdb_filename = fullfile('imdb', sprintf('imdb_%s.mat', opts.data_name));
    if( ~exist(imdb_filename, 'file') )
        make_imdb(imdb_filename, opts);
    end
    fprintf('Load data %s\n', imdb_filename);
    imdb = load(imdb_filename);

    fprintf('Pre-load all images...\n');
    imdb.images.img = batch_imread(imdb.images.filename);

    %% training
    get_batch = @(x,y,mode) getBatch_LapSRN(opts,x,y,mode);

    [net, info] = vllab_cnn_train_dag(model, imdb, get_batch, opts.train, ...
                                      'val', find(imdb.images.set == 2));

