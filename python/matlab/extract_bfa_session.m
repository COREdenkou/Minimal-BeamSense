function out = extract_bfa_session(session_dir, peer_id, varargin)
% extract_bfa_session.m
% Session- and peer-oriented BFA extraction for the Minimal BeamSense hardware loop.
%
% Version adapted for commodity-hardware capture sessions:
%   - Do not assume CSV frame number "No" is identical to parser raw_frame_idx.
%   - Use peer/AP MAC address matching to identify candidate packets.
%   - Use CSV row order as the authoritative packet order.
%   - Carry CSV "Time" forward in the same ordered queue.
%
% Required session contents:
%   session_dir/
%       capture.pcapng
%       session_manifest.json
%       capture_<peer_id>.csv
%
% Output:
%   session_dir/peers/<peer_id>/
%       beamf_angles.mat
%       time_vector.mat
%       exclusive_beamf_report.mat
%       meta.mat
%       vtilde_matrices.mat        (optional)
%
% Optional name-value pairs:
%   'SaveVtilde'      : default false
%   'ProgressEvery'   : default 100
%
% Notes:
%   - This script reuses the BeamSense BFA decoding logic
%     (80 MHz / 234 valid subcarriers / phi_bit=9 / psi_bit=7),
%     while adapting packet selection and alignment for this workflow.
%   - CSV row order is treated as the ordered list of target packets for each peer.
%   - Packet selection uses address matching: the peer MAC and AP BSSID must both
%     appear in the packet address fields.
%   - CSV frame number "No" is kept for debugging/reference rather than alignment.

    ip = inputParser;
    ip.addRequired('session_dir', @(x)ischar(x) || isstring(x));
    ip.addRequired('peer_id', @(x)ischar(x) || isstring(x));
    ip.addParameter('SaveVtilde', false, @(x)islogical(x) || isnumeric(x));
    ip.addParameter('ProgressEvery', 100, @(x)isnumeric(x) && isscalar(x) && x > 0);
    ip.parse(session_dir, peer_id, varargin{:});
    cfg = ip.Results;

    session_dir = char(cfg.session_dir);
    peer_id = char(cfg.peer_id);

    if ~isfolder(session_dir)
        error('Session directory not found: %s', session_dir);
    end

    pcap_file = fullfile(session_dir, 'capture.pcapng');
    if ~isfile(pcap_file)
        error('capture.pcapng not found: %s', pcap_file);
    end

    manifest_file = fullfile(session_dir, 'session_manifest.json');
    if ~isfile(manifest_file)
        error('session_manifest.json not found: %s', manifest_file);
    end

    csv_file = fullfile(session_dir, sprintf('capture_%s.csv', peer_id));
    if ~isfile(csv_file)
        error('Peer CSV not found: %s', csv_file);
    end

    out_dir = fullfile(session_dir, 'peers', peer_id);
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    % ---------- Read manifest ----------
    manifest = jsondecode(fileread(manifest_file));
    [peer_mac, ap_bssid] = resolve_peer_and_ap_mac(manifest, peer_id);

    if isempty(peer_mac)
        error('Could not resolve peer MAC for peer_id=%s from session_manifest.json', peer_id);
    end
    if isempty(ap_bssid)
        error('Could not resolve ap_bssid from session_manifest.json');
    end

    % ---------- Read CSV frame numbers and time ----------
    T = readtable(csv_file);

    frame_col = find_column_case_insensitive(T, "No");
    time_col  = find_column_case_insensitive(T, "Time");

    if isempty(frame_col)
        error('CSV must contain column "No": %s', csv_file);
    end
    if isempty(time_col)
        error('CSV must contain column "Time": %s', csv_file);
    end

    raw_frame_numbers = T.(frame_col);
    if ~isnumeric(raw_frame_numbers)
        error('Column "No" in %s must be numeric.', csv_file);
    end
    raw_frame_numbers = double(raw_frame_numbers(:));

    raw_csv_time_values = convert_column_to_double(T.(time_col));
    raw_csv_time_values = raw_csv_time_values(:);

    valid_row_mask = isfinite(raw_frame_numbers) & raw_frame_numbers >= 1 ...
                   & isfinite(raw_csv_time_values);

    target_frame_numbers = raw_frame_numbers(valid_row_mask);
    target_csv_times = raw_csv_time_values(valid_row_mask);

    if isempty(target_frame_numbers)
        error('No valid rows found in %s', csv_file);
    end

    % Preserve CSV row order as authoritative packet order.
    % If CSV frame numbers are not increasing, sort both by frame number
    % to avoid pathological orderings.
    if any(diff(target_frame_numbers) < 0)
        warning('CSV frame numbers are not monotonic increasing in %s; sorting by frame number.', csv_file);
        [target_frame_numbers, sort_idx] = sort(target_frame_numbers, 'ascend');
        target_csv_times = target_csv_times(sort_idx);
    end

    duplicate_count = sum(diff(target_frame_numbers) == 0);
    if duplicate_count > 0
        warning('Found %d duplicate frame numbers in %s', duplicate_count, csv_file);
    end

    num_targets = numel(target_frame_numbers);

    % ---------- Official BeamSense BFA constants ----------
    Nc = 1;              % one spatial stream per feedback user in this path
    phi_number = 2;
    psi_number = 2;
    skip_start = 255;
    skip_start_sample = 33;

    Nr = 3;
    psi_bit = 7;
    phi_bit = psi_bit + 2;

    NSUBC = 256;
    subcarrier_idxs = linspace(1, NSUBC, NSUBC) - NSUBC/2 - 1;
    pilot_subcarriers = [25, 53, 89, 117, 139, 167, 203, 231];
    num_pilots = numel(pilot_subcarriers);

    subcarrier_idxs(252:end) = [];
    subcarrier_idxs(231) = [];
    subcarrier_idxs(203) = [];
    subcarrier_idxs(167) = [];
    subcarrier_idxs(139) = [];
    subcarrier_idxs(128:130) = [];
    subcarrier_idxs(117) = [];
    subcarrier_idxs(89) = [];
    subcarrier_idxs(53) = [];
    subcarrier_idxs(25) = [];
    subcarrier_idxs(1:6) = [];
    NSUBC_VALID = numel(subcarrier_idxs);

    tot_angles = phi_number + psi_number;
    tot_bits   = phi_number * phi_bit + psi_number * psi_bit;
    tot_bytes  = ceil(tot_bits / 8);

    length_angles = NSUBC_VALID * tot_bytes;
    length_report = ((NSUBC_VALID + num_pilots) / 2 + 1) / 2 * Nc;
    payload_length = Nc + length_angles + length_report;

    fprintf('\n=== extract_bfa_session ===\n');
    fprintf('session_dir          : %s\n', session_dir);
    fprintf('peer_id              : %s\n', peer_id);
    fprintf('peer_mac             : %s\n', peer_mac);
    fprintf('ap_bssid             : %s\n', ap_bssid);
    fprintf('pcap_file            : %s\n', pcap_file);
    fprintf('csv_file             : %s\n', csv_file);
    fprintf('csv target rows      : %d\n', num_targets);
    fprintf('out_dir              : %s\n', out_dir);

    % ---------- Scan PCAP ----------
    p = readpcap_beamf();
    p.open(pcap_file, skip_start);

    beamf_angles       = {};
    excl_beamf_reports = {};
    vtilde_matrices    = {};
    parser_time_vector = {};
    csv_time_vector    = {};
    raw_frame_vector   = {};
    csv_row_vector     = {};
    csv_frame_no_vector = {};

    raw_frame_idx         = 0;
    candidate_idx         = 0;
    csv_consumed_idx      = 0;
    accepted_idx          = 0;
    short_count           = 0;
    parse_fail_count      = 0;
    noncandidate_skip_count = 0;
    extra_candidate_count = 0;
    exact_no_match_count  = 0;

    while true
        f = p.next(payload_length, skip_start_sample);
        if isempty(f.payload)
            disp('no more frames');
            break;
        end

        raw_frame_idx = raw_frame_idx + 1;

        % Address-based candidate selection
        addrs = extract_packet_addrs(f);
        if ~packet_matches_peer_ap(addrs, peer_mac, ap_bssid)
            noncandidate_skip_count = noncandidate_skip_count + 1;
            continue;
        end

        candidate_idx = candidate_idx + 1;

        % CSV row order is authoritative for candidate packets
        if csv_consumed_idx >= num_targets
            extra_candidate_count = extra_candidate_count + 1;
            continue;
        end

        csv_consumed_idx = csv_consumed_idx + 1;
        current_csv_time = target_csv_times(csv_consumed_idx);
        current_csv_no   = target_frame_numbers(csv_consumed_idx);

        if raw_frame_idx == current_csv_no
            exact_no_match_count = exact_no_match_count + 1;
        end

        if size(f.payload, 1) < payload_length
            short_count = short_count + 1;
            if short_count <= 5
                warning('Frame too short at raw_frame_idx=%d (csv_row=%d, csv_No=%d)', ...
                    raw_frame_idx, csv_consumed_idx, current_csv_no);
            end
            continue;
        end

        parser_timestamp = NaN;
        try
            if isfield(f, 'header') && isfield(f.header, 'radiotap_header')
                rt = f.header.radiotap_header;
                if isnumeric(rt)
                    rt = rt(:).';
                    if numel(rt) >= 16
                        timestamp_dec = fliplr(rt(9:16));
                        timestamp_bin = de2bi(timestamp_dec, 'left-msb', 8);
                        timestamp_bin = reshape(timestamp_bin.', 1, []);
                        parser_timestamp = bi2de(timestamp_bin, 'left-msb');
                    end
                end
            end
        catch
            % keep NaN
        end

        try
            start_angles = Nc + 1;
            end_angles   = start_angles + length_angles - 1;
            angle_values = f.payload(start_angles:end_angles);

            beamforming_angles = zeros(NSUBC_VALID, tot_angles, 'uint16');

            for s_i = 1:NSUBC_VALID
                start_idx = (s_i - 1) * tot_bytes + 1;
                end_idx   = s_i * tot_bytes;

                angles_subc_dec = angle_values(start_idx:end_idx);
                angles_subc     = de2bi(angles_subc_dec, 'right-msb', 8);
                angles_subc     = reshape(angles_subc.', 1, []);

                i_curs = 1;
                num_b = [phi_bit, phi_bit, psi_bit, psi_bit];
                for a_i = 1:tot_angles
                    angle_val = bi2de(angles_subc(i_curs:i_curs + num_b(a_i) - 1), 'right-msb');
                    beamforming_angles(s_i, a_i) = uint16(angle_val);
                    i_curs = i_curs + num_b(a_i);
                end
            end

            start_report = start_angles + length_angles;
            end_report   = start_report + length_report - 1;

            exclusive_beamf_report = [];
            if end_report <= numel(f.payload)
                exclusive_beamf_report = f.payload(start_report:end_report);
            end

            accepted_idx = accepted_idx + 1;
            beamf_angles{accepted_idx, 1}         = beamforming_angles;
            excl_beamf_reports{accepted_idx, 1}   = exclusive_beamf_report;
            parser_time_vector{accepted_idx, 1}   = parser_timestamp;
            csv_time_vector{accepted_idx, 1}      = current_csv_time;
            raw_frame_vector{accepted_idx, 1}     = raw_frame_idx;
            csv_row_vector{accepted_idx, 1}       = csv_consumed_idx;
            csv_frame_no_vector{accepted_idx, 1}  = current_csv_no;

            if cfg.SaveVtilde
                try
                    vtilde_matrices{accepted_idx, 1} = Vtilde_NSS1(beamforming_angles, Nc, Nr, NSUBC_VALID, phi_bit, psi_bit);
                catch MEv
                    warning('Vtilde reconstruction failed at accepted_idx=%d: %s', accepted_idx, MEv.message);
                    vtilde_matrices{accepted_idx, 1} = [];
                end
            end

            if mod(accepted_idx, cfg.ProgressEvery) == 0
                fprintf('[INFO] accepted %d candidate frames so far...\n', accepted_idx);
            end

        catch ME
            parse_fail_count = parse_fail_count + 1;
            if parse_fail_count <= 5
                warning('Parse failed at raw_frame_idx=%d (csv_row=%d, csv_No=%d): %s', ...
                    raw_frame_idx, csv_consumed_idx, current_csv_no, ME.message);
            end
            continue;
        end
    end

    % ---------- Save outputs ----------
    beamf_angles_file = fullfile(out_dir, 'beamf_angles.mat');
    time_vector_file  = fullfile(out_dir, 'time_vector.mat');
    excl_file         = fullfile(out_dir, 'exclusive_beamf_report.mat');
    meta_file         = fullfile(out_dir, 'meta.mat');
    vtilde_file       = fullfile(out_dir, 'vtilde_matrices.mat');

    save(beamf_angles_file, 'beamf_angles', '-v7.3');

    time_vector = struct();
    time_vector.parser_time = parser_time_vector;
    time_vector.csv_time = csv_time_vector;
    time_vector.raw_frame_idx = raw_frame_vector;
    time_vector.csv_row_idx = csv_row_vector;
    time_vector.csv_frame_no = csv_frame_no_vector;
    save(time_vector_file, 'time_vector', '-v7.3');

    save(excl_file, 'excl_beamf_reports', '-v7.3');

    if cfg.SaveVtilde
        save(vtilde_file, 'vtilde_matrices', '-v7.3');
    end

    meta = struct();
    meta.version = 'beamsense_v2_reworked_ordinal_match';
    meta.session_dir = session_dir;
    meta.peer_id = peer_id;
    meta.peer_mac = peer_mac;
    meta.ap_bssid = ap_bssid;
    meta.pcap_file = pcap_file;
    meta.csv_file = csv_file;
    meta.manifest_file = manifest_file;

    meta.csv_target_rows = num_targets;
    meta.csv_consumed_rows = csv_consumed_idx;
    meta.accepted_frames = accepted_idx;
    meta.candidate_packets_seen = candidate_idx;
    meta.extra_candidate_packets = extra_candidate_count;
    meta.exact_no_match_count = exact_no_match_count;

    meta.raw_frames_scanned = raw_frame_idx;
    meta.noncandidate_skip_count = noncandidate_skip_count;
    meta.short_frames = short_count;
    meta.parse_fail_count = parse_fail_count;

    meta.Nc = Nc;
    meta.Nr = Nr;
    meta.NSUBC_VALID = NSUBC_VALID;
    meta.phi_bit = phi_bit;
    meta.psi_bit = psi_bit;
    meta.payload_length = payload_length;
    meta.save_vtilde = logical(cfg.SaveVtilde);

    save(meta_file, 'meta', '-v7.3');

    out = struct();
    out.out_dir = out_dir;
    out.peer_id = peer_id;
    out.beamf_angles_file = beamf_angles_file;
    out.time_vector_file = time_vector_file;
    out.exclusive_file = excl_file;
    out.meta_file = meta_file;
    out.accepted_frames = accepted_idx;

    if cfg.SaveVtilde
        out.vtilde_file = vtilde_file;
    else
        out.vtilde_file = '';
    end

    fprintf('[OK] peer %s extracted: accepted_frames=%d, csv_consumed_rows=%d, candidate_packets_seen=%d, exact_no_match_count=%d\n', ...
        peer_id, accepted_idx, csv_consumed_idx, candidate_idx, exact_no_match_count);
end


function [peer_mac, ap_bssid] = resolve_peer_and_ap_mac(manifest, peer_id)
    peer_mac = '';
    ap_bssid = '';

    try
        if isfield(manifest, 'ap_bssid')
            ap_bssid = canonicalize_mac(manifest.ap_bssid);
        end
    catch
        ap_bssid = '';
    end

    try
        if isfield(manifest, 'peer_mapping') && isfield(manifest.peer_mapping, peer_id)
            P = manifest.peer_mapping.(peer_id);

            cand_fields = {'mac', 'peer_mac', 'sta_mac', 'backend_mac'};
            for i = 1:numel(cand_fields)
                f = cand_fields{i};
                if isfield(P, f)
                    peer_mac = canonicalize_mac(P.(f));
                    if ~isempty(peer_mac)
                        break;
                    end
                end
            end
        end
    catch
        peer_mac = '';
    end
end


function addrs = extract_packet_addrs(f)
    addrs = {};

    try
        if isfield(f, 'header')
            H = f.header;

            if isfield(H, 'source_mac')
                m = hexmat_to_macstr(H.source_mac);
                if ~isempty(m), addrs{end+1} = m; end %#ok<AGROW>
            end

            if isfield(H, 'destination_mac')
                m = hexmat_to_macstr(H.destination_mac);
                if ~isempty(m), addrs{end+1} = m; end %#ok<AGROW>
            end

            if isfield(H, 'bss_id')
                m = hexmat_to_macstr(H.bss_id);
                if ~isempty(m), addrs{end+1} = m; end %#ok<AGROW>
            end
        end
    catch
        % keep empty
    end

    if isempty(addrs)
        addrs = {};
        return;
    end

    addrs = cellfun(@canonicalize_mac, addrs, 'UniformOutput', false);
    addrs = addrs(~cellfun(@isempty, addrs));
    if isempty(addrs)
        addrs = {};
    else
        addrs = unique(addrs);
    end
end


function tf = packet_matches_peer_ap(addrs, peer_mac, ap_bssid)
    if isempty(addrs)
        tf = false;
        return;
    end

    has_peer = any(strcmp(addrs, peer_mac));
    has_ap   = any(strcmp(addrs, ap_bssid));
    tf = has_peer && has_ap;
end


function s = hexmat_to_macstr(x)
    s = '';

    try
        if isnumeric(x)
            x = uint8(x(:));
            if numel(x) == 6
                s = sprintf('%02x:%02x:%02x:%02x:%02x:%02x', x(1), x(2), x(3), x(4), x(5), x(6));
                return;
            end
        end

        if ischar(x)
            if size(x, 2) == 2
                parts = cellstr(lower(x));
                s = strjoin(parts(:).', ':');
                return;
            else
                s = canonicalize_mac(x);
                return;
            end
        end

        if isstring(x)
            s = canonicalize_mac(char(x));
            return;
        end
    catch
        s = '';
    end
end


function s = canonicalize_mac(v)
    s = '';
    try
        if isstring(v)
            v = char(v);
        end
        if isnumeric(v)
            if numel(v) == 6
                s = sprintf('%02x:%02x:%02x:%02x:%02x:%02x', v(1), v(2), v(3), v(4), v(5), v(6));
                return;
            else
                return;
            end
        end
        if ~ischar(v)
            return;
        end

        x = lower(v);
        x = regexprep(x, '[^0-9a-f]', '');
        if numel(x) ~= 12
            return;
        end

        parts = cell(1, 6);
        for i = 1:6
            parts{i} = x((i-1)*2 + 1 : i*2);
        end
        s = strjoin(parts, ':');
    catch
        s = '';
    end
end


function name = find_column_case_insensitive(T, target)
    name = '';
    vars = string(T.Properties.VariableNames);
    idx = find(strcmpi(vars, string(target)), 1, 'first');
    if ~isempty(idx)
        name = char(vars(idx));
    end
end


function x = convert_column_to_double(v)
    if isnumeric(v)
        x = double(v);
        return;
    end

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
        return;
    end

    if isstring(v) || ischar(v) || iscategorical(v)
        x = str2double(string(v));
        x = double(x(:));
        return;
    end

    try
        x = double(v);
    catch
        x = nan(numel(v), 1);
    end
end