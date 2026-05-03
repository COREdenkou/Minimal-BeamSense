function out = bfa_to_batches_session(session_dir, peer_id, varargin)
% bfa_to_batches_session.m
% Convert one peer's extracted BFA sequence into fixed BeamSense-style batches.
%
% Input:
%   session_dir/
%       session_manifest.json
%       peers/<peer_id>/
%           beamf_angles.mat
%           time_vector.mat
%           meta.mat
%
% Output:
%   session_dir/peers/<peer_id>/batches/
%       window_000001.mat
%       window_000002.mat
%       ...
%   session_dir/peers/<peer_id>/batches_summary.json
%
% Windowing rule:
%   - interval = 0.1 s
%   - window_size = 10 frames
%   - less than 10 frames => zero pad
%   - more than 10 frames => truncate to first 10
%
% Important:
%   - This workflow uses csv_time as the authoritative time axis.
%   - parser_time fallback is intentionally not used, because parser timestamps
%     may not be reliable for window placement in this capture workflow.
%
% Example:
%   out = bfa_to_batches_session('<SESSION_DIR>', '<PEER_ID>');

    ip = inputParser;
    ip.addRequired('session_dir', @(x)ischar(x) || isstring(x));
    ip.addRequired('peer_id', @(x)ischar(x) || isstring(x));
    ip.addParameter('IntervalSec', 0.1, @(x)isnumeric(x) && isscalar(x) && x > 0);
    ip.addParameter('WindowSize', 10, @(x)isnumeric(x) && isscalar(x) && x >= 1);
    ip.parse(session_dir, peer_id, varargin{:});
    cfg = ip.Results;

    session_dir = char(cfg.session_dir);
    peer_id = char(cfg.peer_id);

    peer_dir = fullfile(session_dir, 'peers', peer_id);
    if ~isfolder(peer_dir)
        error('Peer directory not found: %s', peer_dir);
    end

    manifest_file = fullfile(session_dir, 'session_manifest.json');
    if ~isfile(manifest_file)
        error('session_manifest.json not found: %s', manifest_file);
    end

    beamf_file = fullfile(peer_dir, 'beamf_angles.mat');
    time_file  = fullfile(peer_dir, 'time_vector.mat');
    meta_file  = fullfile(peer_dir, 'meta.mat'); %#ok<NASGU>

    if ~isfile(beamf_file)
        error('beamf_angles.mat not found: %s', beamf_file);
    end
    if ~isfile(time_file)
        error('time_vector.mat not found: %s', time_file);
    end

    manifest = jsondecode(fileread(manifest_file));

    S = load(beamf_file);
    if ~isfield(S, 'beamf_angles')
        error('beamf_angles variable missing in %s', beamf_file);
    end
    beamf_angles = S.beamf_angles;

    T = load(time_file);
    if ~isfield(T, 'time_vector')
        error('time_vector variable missing in %s', time_file);
    end
    time_vector = T.time_vector;

    nFrames = numel(beamf_angles);
    if nFrames == 0
        error('No extracted BFA frames found for peer %s', peer_id);
    end

    frame_times = extract_frame_times(time_vector);

    if numel(frame_times) ~= nFrames
        warning('frame_times length (%d) != beamf_angles length (%d); truncating to min length.', numel(frame_times), nFrames);
        nMin = min(numel(frame_times), nFrames);
        frame_times = frame_times(1:nMin);
        beamf_angles = beamf_angles(1:nMin);
        nFrames = nMin;
    end

    valid_mask = isfinite(frame_times) & frame_times >= 0;
    if nnz(valid_mask) < nFrames
        warning('Dropping %d frames with invalid csv_time for peer %s.', nFrames - nnz(valid_mask), peer_id);
        frame_times = frame_times(valid_mask);
        beamf_angles = beamf_angles(valid_mask);
        nFrames = numel(frame_times);
    end

    if nFrames == 0
        error('No usable frames remain for peer %s after filtering invalid csv_time.', peer_id);
    end

    % Capture duration from manifest if available
    capture_sec = [];
    try
        capture_sec = double(manifest.capture.capture_sec);
    catch
    end
    if isempty(capture_sec) || ~isfinite(capture_sec) || capture_sec <= 0
        capture_sec = max(frame_times);
    end

    interval = cfg.IntervalSec;
    window_size = cfg.WindowSize;
    num_windows = ceil(capture_sec / interval);

    batches_dir = fullfile(peer_dir, 'batches');
    if ~exist(batches_dir, 'dir')
        mkdir(batches_dir);
    end

    informative_windows = 0;
    dense_windows_ge3 = 0;
    full_windows_eq10 = 0;
    valid_counts = zeros(num_windows, 1);

    fprintf('\n=== bfa_to_batches_session ===\n');
    fprintf('session_dir    : %s\n', session_dir);
    fprintf('peer_id        : %s\n', peer_id);
    fprintf('frames         : %d\n', nFrames);
    fprintf('capture_sec    : %.3f\n', capture_sec);
    fprintf('interval       : %.3f\n', interval);
    fprintf('window_size    : %d\n', window_size);
    fprintf('num_windows    : %d\n', num_windows);
    fprintf('time_min       : %.6f\n', min(frame_times));
    fprintf('time_max       : %.6f\n', max(frame_times));

    for w = 1:num_windows
        t0 = (w - 1) * interval;
        t1 = w * interval;

        idx = find(frame_times >= t0 & frame_times < t1);
        valid_n = numel(idx);
        valid_counts(w) = valid_n;

        if valid_n > 0
            informative_windows = informative_windows + 1;
        end
        if valid_n >= 3
            dense_windows_ge3 = dense_windows_ge3 + 1;
        end
        if valid_n >= window_size
            full_windows_eq10 = full_windows_eq10 + 1;
        end

        bf_matrix = zeros(window_size, 234, 4);

        nUse = min(valid_n, window_size);
        for k = 1:nUse
            this_bfa = beamf_angles{idx(k)};
            % Expected shape: 234 x 4
            if ~isequal(size(this_bfa), [234, 4])
                error('Unexpected BFA shape at window %d frame %d: got [%d x %d], expected [234 x 4].', ...
                    w, k, size(this_bfa,1), size(this_bfa,2));
            end
            bf_matrix(k, :, :) = this_bfa;
        end

        window_index = w;
        window_start = t0;
        window_end = t1;
        valid_frames = valid_n;

        save(fullfile(batches_dir, sprintf('window_%06d.mat', w)), ...
            'bf_matrix', 'window_index', 'window_start', 'window_end', 'valid_frames', 'peer_id', '-v7.3');
    end

    summary = struct();
    summary.version = 'beamsense_v2';
    summary.session_dir = session_dir;
    summary.peer_id = peer_id;
    summary.num_windows = num_windows;
    summary.interval_sec = interval;
    summary.window_size = window_size;
    summary.capture_sec = capture_sec;
    summary.num_input_frames = nFrames;
    summary.informative_windows = informative_windows;
    summary.dense_windows_ge3 = dense_windows_ge3;
    summary.full_windows_eq10 = full_windows_eq10;
    summary.mean_valid_frames = mean(valid_counts);
    summary.median_valid_frames = median(valid_counts);
    summary.time_min = min(frame_times);
    summary.time_max = max(frame_times);

    summary_json_file = fullfile(peer_dir, 'batches_summary.json');
    fid = fopen(summary_json_file, 'w');
    if fid == -1
        error('Failed to open summary json for writing: %s', summary_json_file);
    end
    fwrite(fid, jsonencode(summary, 'PrettyPrint', true), 'char');
    fclose(fid);

    out = struct();
    out.batches_dir = batches_dir;
    out.summary_json = summary_json_file;
    out.num_windows = num_windows;
    out.informative_windows = informative_windows;
    out.mean_valid_frames = summary.mean_valid_frames;

    fprintf('[OK] peer %s batches written: %s\n', peer_id, batches_dir);
end


function frame_times = extract_frame_times(time_vector)
    % Only trust csv_time.
    if isfield(time_vector, 'csv_time')
        frame_times = cell_or_array_to_double(time_vector.csv_time);
        frame_times = sanitize_relative_times(frame_times);
        if nnz(isfinite(frame_times)) >= 3
            return;
        end
    end

    error(['csv_time unavailable or invalid; parser_time fallback is disabled because ' ...
           'parser timestamps are currently unreliable for window placement.']);
end


function x = cell_or_array_to_double(v)
    if iscell(v)
        x = nan(numel(v), 1);
        for i = 1:numel(v)
            try
                vi = v{i};
                if isempty(vi)
                    x(i) = NaN;
                elseif isnumeric(vi)
                    x(i) = double(vi(1));
                else
                    x(i) = str2double(string(vi));
                end
            catch
                x(i) = NaN;
            end
        end
    else
        x = double(v(:));
    end
end


function t = sanitize_relative_times(t)
    % Preserve vector length; only normalize, do not silently compress.
    t = t(:);

    finite_mask = isfinite(t) & t >= 0;
    if ~any(finite_mask)
        return;
    end

    finite_vals = t(finite_mask);

    % If timestamps look absolute-ish, convert to relative by subtracting first valid value.
    if finite_vals(1) > 1e3
        t(finite_mask) = finite_vals - finite_vals(1);
    end

    % Warn on non-monotonicity, but do not reorder automatically.
    finite_vals2 = t(finite_mask);
    if any(diff(finite_vals2) < -1e-9)
        warning('csv_time is non-monotonic after normalization; downstream windowing may be affected.');
    end
end