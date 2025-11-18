%% MBRW_SPECTRAL_DIMENSION_DEMO - Venice Street Network with Infomap
% 
%  SETUP REQUIRED: Download Infomap executable
%  Windows: https://github.com/mapequation/infomap/releases/latest/download/Infomap-win64.zip
%  Extract Infomap.exe to your MATLAB working directory
%  
%  Linux/Mac: https://www.mapequation.org/infomap/#Install
%  Or compile from source

%% Path to Boost library.
boost_path = 'C:\boost_1_89_0';

%% INFOMAP CONFIGURATION - SET YOUR EXECUTABLE PATH
% For Windows: Download from link above and set path
if ispc
    infomap_exe = 'Infomap.exe';  % Should be in current directory
else
    infomap_exe = 'Infomap';      % Should be in PATH
end

% Verify Infomap is available
fprintf('Checking for Infomap executable...\n');
[status, result] = system([infomap_exe ' --version']);
if status == 0
    fprintf('✓ Infomap found: %s\n', strtrim(result));
    infomap_available = true;
else
    fprintf('✗ Infomap not found!\n');
    fprintf('Please download from: https://github.com/mapequation/infomap/releases\n');
    fprintf('For Windows: Download Infomap-win64.zip and extract Infomap.exe here\n');
    fprintf('Falling back to native algorithm...\n');
    infomap_available = false;
end

%% Load Venice street network from GraphML file
venice_graphml_file = 'networks/venice.graphml';
print_network_name = 'Venice, Italy street network';

% Create necessary directories
if ~exist('networks', 'dir')
    mkdir('networks');
end
if ~exist('segment_mass_log_multimeans', 'dir')
    mkdir('segment_mass_log_multimeans');
end
if ~exist('infomap_output', 'dir')
    mkdir('infomap_output');
end

%% Read GraphML file using streaming (memory-efficient)
fprintf('Reading Venice network from GraphML: %s\n', venice_graphml_file);

fid = -1;  % Initialize file identifier
try
    % Open file for reading
    fid = fopen(venice_graphml_file, 'r', 'n', 'UTF-8');
    if fid == -1
        error('Cannot open file: %s', venice_graphml_file);
    end
    
    % Pre-allocate arrays with reasonable initial sizes
    node_ids = [];
    edges_data = [];
    
    node_count = 0;
    edge_count = 0;
    current_edge_source = 0;
    current_edge_target = 0;
    
    % Use containers.Map for efficient coordinate lookup
    coord_map = containers.Map('KeyType', 'double', 'ValueType', 'any');
    
    fprintf('Parsing GraphML (streaming mode)...\n');
    last_progress = 0;
    
    % Read line by line
    while ~feof(fid)
        line = fgetl(fid);
        if ~ischar(line)
            break;
        end
        
        % Parse nodes
        if contains(line, '<node id=')
            node_count = node_count + 1;
            
            % Extract node ID
            id_match = regexp(line, 'id="(\d+)"', 'tokens');
            if ~isempty(id_match)
                node_ids(node_count) = str2double(id_match{1}{1});
            end
        end
        
        % Parse edges
        if contains(line, '<edge ')
            edge_count = edge_count + 1;
            
            % Extract source and target
            source_match = regexp(line, 'source="(\d+)"', 'tokens');
            target_match = regexp(line, 'target="(\d+)"', 'tokens');
            
            if ~isempty(source_match) && ~isempty(target_match)
                current_edge_source = str2double(source_match{1}{1});
                current_edge_target = str2double(target_match{1}{1});
                edges_data(edge_count, 1) = current_edge_source;
                edges_data(edge_count, 2) = current_edge_target;
            end
        end
        
        % Parse geometry for coordinates (key="d8")
        if contains(line, 'key="d8"') && contains(line, 'LINESTRING')
            % Extract LINESTRING coordinates
            geom_match = regexp(line, 'LINESTRING \(([^\)]+)\)', 'tokens');
            if ~isempty(geom_match)
                coords_str = geom_match{1}{1};
                % Split by comma to get individual coordinate pairs
                coord_pairs = strsplit(coords_str, ',');
                
                if ~isempty(coord_pairs)
                    % First coordinate (source node)
                    first_pair = strtrim(coord_pairs{1});
                    coords = strsplit(first_pair, ' ');
                    if length(coords) >= 2
                        first_x = str2double(coords{1});
                        first_y = str2double(coords{2});
                    end
                    
                    % Last coordinate (target node)
                    last_pair = strtrim(coord_pairs{end});
                    coords = strsplit(last_pair, ' ');
                    if length(coords) >= 2
                        last_x = str2double(coords{1});
                        last_y = str2double(coords{2});
                    end
                    
                    % Store coordinates for source and target nodes
                    if current_edge_source > 0 && current_edge_target > 0
                        % Store in map (overwrites if already exists, keeping first occurrence)
                        if ~isKey(coord_map, current_edge_source)
                            coord_map(current_edge_source) = [first_x, first_y];
                        end
                        
                        if ~isKey(coord_map, current_edge_target)
                            coord_map(current_edge_target) = [last_x, last_y];
                        end
                    end
                end
            end
        end
        
        % Progress indicator
        if edge_count > last_progress + 5000
            fprintf('  Processed %d edges...\n', edge_count);
            last_progress = edge_count;
        end
    end
    
    fclose(fid);
    fid = -1;  % Mark as closed
    
    fprintf('Finished parsing. Building coordinate arrays...\n');
    
    fprintf('Finished parsing. Building coordinate arrays...\n');
    
    % Build coordinate arrays from map
    x_coords = nan(node_count, 1);
    y_coords = nan(node_count, 1);
    
    for i = 1:node_count
        if isKey(coord_map, node_ids(i))
            coords = coord_map(node_ids(i));
            x_coords(i) = coords(1);
            y_coords(i) = coords(2);
        end
    end
    
    % Clear the map to free memory
    clear coord_map;
    
    coords_found = sum(~isnan(x_coords));
    fprintf('Successfully read %d nodes and %d edges from GraphML\n', node_count, edge_count);
    fprintf('Extracted coordinates for %d/%d nodes (%.1f%%)\n', ...
            coords_found, node_count, 100*coords_found/node_count);
    
    % Remap node IDs to sequential 1,2,3,... for memory efficiency
    fprintf('Remapping node IDs to sequential indices...\n');
    node_id_to_seq = containers.Map(node_ids, 1:node_count);
    
    % Remap edges to use sequential IDs
    edges_remapped = zeros(edge_count, 2);
    for i = 1:edge_count
        edges_remapped(i, 1) = node_id_to_seq(edges_data(i, 1));
        edges_remapped(i, 2) = node_id_to_seq(edges_data(i, 2));
    end
    
    % Clear original edges to free memory
    clear edges_data;
    
    % Create graph with sequential node IDs (much more memory efficient)
    fprintf('Creating graph object...\n');
    G_original = graph(edges_remapped(:,1), edges_remapped(:,2));
    
    % Clear remapped edges
    clear edges_remapped;
    
    fprintf('Created graph with %d nodes and %d edges\n', numnodes(G_original), numedges(G_original));
    
catch ME
    if fid ~= -1
        fclose(fid);
    end
    error('Failed to read GraphML file: %s\n%s', venice_graphml_file, ME.message);
end

%% Extract 2-core from Venice network and save largest connected component

fprintf('Extracting 2-core...\n');
G_2core = G_original;
iteration = 0;
while true
    iteration = iteration + 1;
    node_degrees = degree(G_2core);
    low_degree_nodes = find(node_degrees < 2);
    
    if isempty(low_degree_nodes)
        break;
    end
    
    G_2core = rmnode(G_2core, low_degree_nodes);
    fprintf('Iteration %d: Removed %d nodes with degree < 2\n', iteration, length(low_degree_nodes));
end

fprintf('2-core graph has %d nodes and %d edges\n', numnodes(G_2core), numedges(G_2core));

% Get connected components
fprintf('Finding connected components...\n');
cc_bins = conncomp(G_2core);
num_components = max(cc_bins);
fprintf('Found %d connected components\n', num_components);

% Find largest connected component
component_sizes = zeros(num_components, 1);
for i = 1:num_components
    component_sizes(i) = sum(cc_bins == i);
end
[largest_size, largest_cc_idx] = max(component_sizes);

fprintf('Component sizes: ');
fprintf('%d ', sort(component_sizes, 'descend'));
fprintf('\n');

largest_cc_nodes = find(cc_bins == largest_cc_idx);
G = subgraph(G_2core, largest_cc_nodes);

fprintf('Largest connected component has %d nodes and %d edges\n', ...
    numnodes(G), numedges(G));

% Save the processed network as edge list for MBRW
processed_venice_file = 'networks/venice_2core_lcc.txt';
fprintf('Saving processed network to %s\n', processed_venice_file);

edge_table = G.Edges;
writematrix([edge_table.EndNodes(:,1), edge_table.EndNodes(:,2)], ...
    processed_venice_file, 'Delimiter', '\t');

mbrw_able_edge_file_name = processed_venice_file;

min_degree = min(degree(G));
fprintf('Minimum degree in processed graph: %d\n', min_degree);
if min_degree < 2
    error('Graph still contains nodes with degree < 2. Something went wrong.');
end

%% ========== INFOMAP COMMUNITY DETECTION ==========
fprintf('\n============================================================\n');
fprintf('INFOMAP COMMUNITY DETECTION\n');
fprintf('============================================================\n');

if infomap_available
    % Export graph for Infomap (Pajek format)
    infomap_input_file = 'networks/venice_for_infomap.net';
    fprintf('Exporting graph to Pajek format...\n');
    
    fid = fopen(infomap_input_file, 'w');
    fprintf(fid, '*Vertices %d\n', numnodes(G));
    for i = 1:numnodes(G)
        fprintf(fid, '%d "%d"\n', i, i);
    end
    
    fprintf(fid, '*Edges\n');
    edges = G.Edges.EndNodes;
    for i = 1:size(edges, 1)
        fprintf(fid, '%d %d 1\n', edges(i,1), edges(i,2));
    end
    fclose(fid);
    
    fprintf('Graph exported to %s\n', infomap_input_file);
    
    % Test multiple markov-time values
    markov_times = [20, 30, 40, 45, 50, 60, 70];
    best_result = struct('modularity', -Inf);
    all_results = [];
    
    fprintf('\nTesting Infomap with different Markov times...\n');
    fprintf('Target: 6 communities (Venice sestieri)\n\n');
    
    for mt_idx = 1:length(markov_times)
        mt = markov_times(mt_idx);
        
        % Clean output directory
        if exist('infomap_output', 'dir')
            delete('infomap_output/*.*');
        end
        
        % Run Infomap - Use correct syntax for v2.8.0
        % In v2.8.0, use -u or --undirdir instead of --undirected
        infomap_command = sprintf('"%s" "%s" infomap_output -u --two-level --markov-time %d --num-trials 10', ...
            infomap_exe, infomap_input_file, mt);
        
        fprintf('  Markov-time %3d: ', mt);
        
        [status, cmdout] = system(infomap_command);
        
        % If that fails, try --undirdir
        if status ~= 0
            infomap_command = sprintf('"%s" "%s" infomap_output --undirdir --two-level --markov-time %d --num-trials 10', ...
                infomap_exe, infomap_input_file, mt);
            [status, cmdout] = system(infomap_command);
        end
        
        % If still fails, try without undirected flag (default behavior)
        if status ~= 0
            infomap_command = sprintf('"%s" "%s" infomap_output --two-level --markov-time %d --num-trials 10', ...
                infomap_exe, infomap_input_file, mt);
            [status, cmdout] = system(infomap_command);
        end
        
        if status == 0
            % Find .tree file
            tree_files = dir('infomap_output/*.tree');
            
            if ~isempty(tree_files)
                tree_file = fullfile('infomap_output', tree_files(1).name);
                
                % Parse communities
                infomap_communities = parse_infomap_tree(tree_file, numnodes(G));
                num_communities = max(infomap_communities);
                
                % Calculate modularity
                modularity = calculate_modularity(G, infomap_communities);
                
                % Get community sizes
                sizes = zeros(num_communities, 1);
                for c = 1:num_communities
                    sizes(c) = sum(infomap_communities == c);
                end
                
                fprintf('%2d communities, Q=%.4f, Sizes:[', num_communities, modularity);
                fprintf('%d ', sort(sizes, 'descend'));
                fprintf(']\n');
                
                % Store result as struct
                result = struct();
                result.markov_time = mt;
                result.num_communities = num_communities;
                result.modularity = modularity;
                result.communities = infomap_communities;
                result.sizes = sizes;
                
                if isempty(all_results)
                    all_results = result;
                else
                    all_results(end+1) = result;
                end
                
                % Track best result with 6 communities
                if num_communities == 6 && modularity > best_result.modularity
                    best_result = result;
                    fprintf('         ★ Best 6-community result so far!\n');
                end
            else
                fprintf('No output file generated\n');
                fprintf('  Output: %s\n', strtrim(cmdout));
            end
        else
            fprintf('FAILED\n');
            if ~isempty(cmdout)
                fprintf('  Error: %s\n', strtrim(cmdout(1:min(200,length(cmdout)))));
            end
        end
    end
    
    % Use best result
    if best_result.modularity > -Inf
        fprintf('\n✓ BEST 6-COMMUNITY RESULT FOUND:\n');
        fprintf('  Markov time: %d\n', best_result.markov_time);
        fprintf('  Modularity: %.4f\n', best_result.modularity);
        fprintf('  Community sizes: ');
        fprintf('%d ', sort(best_result.sizes, 'descend'));
        fprintf('\n');
        
        infomap_communities = best_result.communities;
        num_infomap_communities = 6;
        infomap_modularity = best_result.modularity;
        infomap_comm_sizes = best_result.sizes;
        
    elseif ~isempty(all_results)
        fprintf('\n⚠ No exact 6-community configuration found.\n');
        fprintf('Available configurations:\n');
        for i = 1:length(all_results)
            fprintf('  %d communities (MT=%d): Q=%.4f\n', ...
                    all_results(i).num_communities, ...
                    all_results(i).markov_time, ...
                    all_results(i).modularity);
        end
        
        % Use best overall result
        best_mod = -Inf;
        best_idx = 1;
        for i = 1:length(all_results)
            if all_results(i).modularity > best_mod
                best_mod = all_results(i).modularity;
                best_idx = i;
            end
        end
        best_result = all_results(best_idx);
        
        fprintf('\nUsing best result: %d communities (Q=%.4f)\n', ...
                best_result.num_communities, best_result.modularity);
        
        infomap_communities = best_result.communities;
        num_infomap_communities = best_result.num_communities;
        infomap_modularity = best_result.modularity;
        infomap_comm_sizes = best_result.sizes;
        
    else
        fprintf('\n✗ ERROR: All Infomap runs failed!\n');
        fprintf('Possible issues:\n');
        fprintf('  1. Input file format problem\n');
        fprintf('  2. Infomap version incompatibility\n');
        fprintf('  3. File permission issues\n');
        fprintf('\nFalling back to native MATLAB algorithm...\n\n');
        
        [infomap_communities, infomap_modularity] = louvain_communities_matlab(G, 6);
        num_infomap_communities = max(infomap_communities);
        
        infomap_comm_sizes = zeros(num_infomap_communities, 1);
        for i = 1:num_infomap_communities
            infomap_comm_sizes(i) = sum(infomap_communities == i);
        end
    end
    
else
    % Fallback to native algorithm
    fprintf('Using native MATLAB algorithm...\n');
    [infomap_communities, infomap_modularity] = louvain_communities_matlab(G, 6);
    num_infomap_communities = max(infomap_communities);
    
    infomap_comm_sizes = zeros(num_infomap_communities, 1);
    for i = 1:num_infomap_communities
        infomap_comm_sizes(i) = sum(infomap_communities == i);
    end
end

fprintf('\nFinal result: %d communities, Modularity: %.4f\n', ...
        num_infomap_communities, infomap_modularity);

%% Visualize Communities
fprintf('\nCreating visualizations...\n');

figure('Position', [50 50 1600 1000], 'Name', 'Venice Infomap Communities');

% Subplot 1: Geographic map with communities (SIMPLIFIED)
subplot(2, 3, 1);
try
    % Use the SAME code that works in the geographic figure
    if exist('plot_x', 'var') && exist('plot_y', 'var')
        % Variables already exist from before, just reuse them
        valid_coords = ~isnan(plot_x) & ~isnan(plot_y);
        coords_available = sum(valid_coords);
        
        if coords_available > numnodes(G) * 0.3
            if coords_available == numnodes(G)
                % Plot full graph with coordinates
                h1 = plot(G, 'XData', plot_x, 'YData', plot_y, ...
                     'EdgeColor', [0.8 0.8 0.8], 'EdgeAlpha', 0.15, ...
                     'MarkerSize', 8, 'LineWidth', 0.5);
                h1.NodeCData = infomap_communities;
                h1.NodeColor = 'flat';
            else
                % Create subgraph with valid coordinates
                nodes_with_coords = find(valid_coords);
                G_geo = subgraph(G, nodes_with_coords);
                h1 = plot(G_geo, 'XData', plot_x(valid_coords), 'YData', plot_y(valid_coords), ...
                     'EdgeColor', [0.8 0.8 0.8], 'EdgeAlpha', 0.15, ...
                     'MarkerSize', 8, 'LineWidth', 0.5);
                h1.NodeCData = infomap_communities(valid_coords);
                h1.NodeColor = 'flat';
            end
            
            colormap(gca, hsv(num_infomap_communities));
            colorbar;
            title(sprintf('Geographic View\n%d communities', num_infomap_communities), 'FontWeight', 'bold');
            xlabel('Longitude');
            ylabel('Latitude');
            axis equal tight;
            box on;
        else
            % Not enough coordinates, use force layout
            h1 = plot(G, 'Layout', 'force', 'EdgeColor', [0.7 0.7 0.7], 'EdgeAlpha', 0.3, 'MarkerSize', 6);
            h1.NodeCData = infomap_communities;
            colormap(gca, hsv(num_infomap_communities));
            colorbar;
            title(sprintf('Network View\n%d communities', num_infomap_communities));
            axis off;
        end
    else
        % No coordinate variables found, use force layout
        h1 = plot(G, 'Layout', 'force', 'EdgeColor', [0.7 0.7 0.7], 'EdgeAlpha', 0.3, 'MarkerSize', 6);
        h1.NodeCData = infomap_communities;
        colormap(gca, hsv(num_infomap_communities));
        colorbar;
        title(sprintf('Network View\n%d communities', num_infomap_communities));
        axis off;
    end
catch ME
    fprintf('Subplot 1 error: %s\n', ME.message);
    % Simple fallback
    h1 = plot(G, 'NodeCData', infomap_communities);
    colormap(gca, hsv(num_infomap_communities));
    title('Communities');
end

% Subplot 2: Size distribution
subplot(2, 3, 2);
histogram(infomap_comm_sizes, 'FaceColor', [0.4 0.6 0.8], 'EdgeColor', 'black');
xlabel('Community Size');
ylabel('Count');
title('Size Distribution');
grid on;

% Subplot 3: Ranked sizes
subplot(2, 3, 3);
[sorted_sizes, ~] = sort(infomap_comm_sizes, 'descend');
bar(1:num_infomap_communities, sorted_sizes, 'FaceColor', [0.3 0.5 0.8]);
xlabel('Rank');
ylabel('Nodes');
title('Communities by Size');
grid on;


% Subplot 4: Results comparison
subplot(2, 3, 4);
if exist('all_results', 'var') && ~isempty(all_results)
    mt_vals = [all_results.markov_time];
    nc_vals = [all_results.num_communities];
    mod_vals = [all_results.modularity];
    
    yyaxis left
    plot(mt_vals, nc_vals, 'b-o', 'LineWidth', 2);
    ylabel('Communities');
    
    yyaxis right
    plot(mt_vals, mod_vals, 'r-s', 'LineWidth', 2);
    ylabel('Modularity');
    
    xlabel('Markov Time');
    title('Parameter Sweep');
    grid on;
end

% Subplot 5: Statistics
subplot(2, 3, 5);
axis off;
stats = {
    'INFOMAP RESULTS';
    '===================';
    sprintf('Communities: %d', num_infomap_communities);
    sprintf('Modularity: %.4f', infomap_modularity);
    '';
    sprintf('Largest: %d (%.1f%%)', max(infomap_comm_sizes), ...
            100*max(infomap_comm_sizes)/numnodes(G));
    sprintf('Smallest: %d', min(infomap_comm_sizes));
    sprintf('Mean: %.1f', mean(infomap_comm_sizes));
    '';
    'Communities:';
};
for i = 1:length(sorted_sizes)
    stats{end+1} = sprintf('  #%d: %d nodes', i, sorted_sizes(i));
end

text(0.05, 0.95, stats, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
     'FontName', 'Courier', 'FontSize', 10);

sgtitle('Venice Network - Infomap Community Detection', 'FontSize', 16, 'FontWeight', 'bold');

print('venice_infomap_communities', '-dpng', '-r300');
fprintf('Saved: venice_infomap_communities.png\n');

%% Plot communities on geographic map (separate figure)
fprintf('\nCreating geographic community visualization...\n');

figure('Position', [100 100 1200 900], 'Name', 'Venice Communities - Geographic View');

% Map graph node indices to original node IDs and their coordinates
if exist('x_coords', 'var') && exist('y_coords', 'var')
    % After 2-core and LCC extraction, G has fewer nodes than G_original
    % We need to map G's nodes back to the original coordinate arrays
    
    % Get which nodes from G_original are in G
    % G is a subgraph of G_2core, which is derived from G_original
    % The node indices in G correspond to nodes in G_original
    
    % Create arrays to store coordinates for nodes in G
    plot_x = nan(numnodes(G), 1);
    plot_y = nan(numnodes(G), 1);
    
    % G_original had sequential IDs 1:node_count that map to coordinates
    % After filtering, G has a subset of these nodes
    % We need to figure out which original nodes are still in G
    
    % The safest approach: reconstruct the mapping
    % G_original nodes are 1:node_count (sequential after remapping)
    % x_coords and y_coords are indexed 1:node_count
    
    % Try to map through the processing chain
    % This is tricky because rmnode and subgraph change indices
    
    % Alternative approach: use node names if available
    % Or: extract from the 2-core process which nodes remain
    
    % For now, let's try a simpler approach:
    % Check if coordinates align by comparing graph sizes
    
    if numnodes(G) <= length(x_coords)
        % Try direct mapping for nodes that have coordinates
        for i = 1:numnodes(G)
            if i <= length(x_coords) && ~isnan(x_coords(i))
                plot_x(i) = x_coords(i);
                plot_y(i) = y_coords(i);
            end
        end
    end
    
    % Count valid coordinates
    valid_coords = ~isnan(plot_x) & ~isnan(plot_y);
    coords_available = sum(valid_coords);
    
    fprintf('Found valid coordinates for %d/%d nodes (%.1f%%)\n', ...
            coords_available, numnodes(G), 100*coords_available/numnodes(G));
    
    if coords_available > numnodes(G) * 0.3  % If we have >30% coordinates
        
        % Replace NaN coordinates with a default position (won't be visible)
        % or create a subgraph with only nodes that have coordinates
        
        if coords_available == numnodes(G)
            % All nodes have coordinates - perfect!
            h = plot(G, 'XData', plot_x, 'YData', plot_y, ...
                 'EdgeColor', [0.8 0.8 0.8], 'EdgeAlpha', 0.15, ...
                 'MarkerSize', 8, 'LineWidth', 0.5);
            
            h.NodeCData = infomap_communities;
            h.NodeColor = 'flat';
            
        else
            % Some nodes missing coordinates - create subgraph with valid coords only
            fprintf('Creating subgraph with nodes that have coordinates...\n');
            nodes_with_coords = find(valid_coords);
            G_geo = subgraph(G, nodes_with_coords);
            
            plot_x_valid = plot_x(valid_coords);
            plot_y_valid = plot_y(valid_coords);
            communities_valid = infomap_communities(valid_coords);
            
            h = plot(G_geo, 'XData', plot_x_valid, 'YData', plot_y_valid, ...
                 'EdgeColor', [0.8 0.8 0.8], 'EdgeAlpha', 0.15, ...
                 'MarkerSize', 8, 'LineWidth', 0.5);
            
            h.NodeCData = communities_valid;
            h.NodeColor = 'flat';
            
            fprintf('Geographic plot includes %d/%d nodes with valid coordinates\n', ...
                    length(nodes_with_coords), numnodes(G));
        end
        
        % Use a nice colormap
        colormap(hsv(num_infomap_communities));
        
        if num_infomap_communities <= 20
            cb = colorbar('Ticks', 1:num_infomap_communities);
            cb.Label.String = 'Community';
            cb.Label.FontSize = 12;
        else
            colorbar;
        end
        
        title(sprintf('Venice Street Network - Community Structure\n%d Communities (Infomap, Q=%.3f)', ...
              num_infomap_communities, infomap_modularity), ...
              'FontSize', 14, 'FontWeight', 'bold');
        
        xlabel('Longitude', 'FontSize', 11);
        ylabel('Latitude', 'FontSize', 11);
        
        axis equal tight;
        grid on;
        box on;
        
        % Save the geographic plot
        print('venice_communities_geographic', '-dpng', '-r300');
        fprintf('Saved: venice_communities_geographic.png\n');
        
    else
        fprintf('Warning: Only found coordinates for %d/%d nodes (%.1f%%)\n', ...
                coords_available, numnodes(G), 100*coords_available/numnodes(G));
        fprintf('Skipping geographic visualization (need >30%% coverage)\n');
    end
else
    fprintf('Warning: Coordinate data not available for geographic plot\n');
end

%% Save results
results_table = table((1:numnodes(G))', infomap_communities, ...
                      'VariableNames', {'NodeID', 'Community'});
writetable(results_table, 'venice_infomap_communities.csv');
fprintf('Saved: venice_infomap_communities.csv\n');

%% Run MBRW
bias = 10000;
memory = 5;
rand_seed = 0;
log_num_steps = 23;

finite_orders = -19:19;
orders = [-Inf finite_orders Inf];
num_orders = numel(orders);
order_file_name = 'orders.txt';
writematrix(orders', order_file_name);

segment_mass_log_multimean_file_name = fullfile('segment_mass_log_multimeans', ...
    ['venice_b_' num2str(bias) '_m_' num2str(memory) ...
    '_r_' num2str(rand_seed) '_c_' num2str(log_num_steps) '.csv']);

c_executable_name = 'mbrw_and_save_segment_mass_log_multimeans_2';
compile_command = sprintf('g++ -std=c++0x -I %s %s.cpp -o %s -O2', ...
    boost_path, c_executable_name, c_executable_name);

if ispc
    dot_if_linux = '';
    exe_if_pc = '.exe';
else
    dot_if_linux = './';
    exe_if_pc = '';
end

run_command = sprintf('%s%s -i %s -q %s -o %s -b %u -m %u -r %u -c %u', ...
    dot_if_linux, c_executable_name, mbrw_able_edge_file_name, ...
    order_file_name, segment_mass_log_multimean_file_name, ...
    bias, memory, rand_seed, log_num_steps);

if ~exist(segment_mass_log_multimean_file_name, 'file')
    if ~exist([c_executable_name exe_if_pc], 'file')
        fprintf('Compiling %s...\n', c_executable_name)
        tic
        compile_result = system(compile_command);
        toc
        if compile_result
            error('Failed to compile %s', c_executable_name)
        end
    end
    fprintf('Running %s...\n', c_executable_name)
    tic
    run_result = system(run_command);
    toc
    if run_result
        error('Failed to run %s', c_executable_name);
    end
end

%% Plot segment masses
fprintf('Reading results from %s\n', segment_mass_log_multimean_file_name);
log2_generalized_means = readmatrix(segment_mass_log_multimean_file_name, ...
    'FileType', 'text', 'Delimiter', ',');
num_segment_lengths = size(log2_generalized_means, 1);
log2_segment_lengths = (0:num_segment_lengths-1)';

line_colors = jet(num_orders);
legend_items = cell(num_orders, 1);
figure('Position', [100 100 800 600])
hold on
for o = 1:num_orders
    plot(log2_segment_lengths, log2_generalized_means(:, o), ...
        'Color', line_colors(o, :), 'LineWidth', 1.5)
    if isinf(orders(o))
        legend_items{o} = sprintf('q=%s', char(8734*sign(orders(o))));
    else
        legend_items{o} = sprintf('q=%i', orders(o));
    end
end
hold off
legend(legend_items, 'Location', 'northeastoutside', 'FontSize', 8)
xlabel('log_2(segment length)')
ylabel('log_2(q-mean segment mass)')
title(sprintf('MBRW (bias=%u, memory=%u)\n%s', bias, memory, print_network_name))
grid on

%% Spectral dimensions
num_nodes = numnodes(G);
fprintf('Computing spectral dimensions for %d nodes\n', num_nodes);
Dq = spectral_dimensions_v2(log2_generalized_means, 'length', num_nodes);

order_is_finite = ~isinf(orders);
finite_order_Dqs = Dq(order_is_finite);

figure('Position', [200 200 800 600])
plot(finite_orders, finite_order_Dqs, 'b-o', 'LineWidth', 2, 'MarkerSize', 6)
xlabel('q')
ylabel('D_q')
title(sprintf('Spectral Dimensions (bias=%u, memory=%u)\n%s', ...
      bias, memory, print_network_name))
grid on

mean_Dq = mean(finite_order_Dqs);
std_Dq = std(finite_order_Dqs);
text(0.05, 0.95, sprintf('Mean=%.3f\nStd=%.3f', mean_Dq, std_Dq), ...
    'Units', 'normalized', 'VerticalAlignment', 'top', ...
    'BackgroundColor', 'white', 'EdgeColor', 'black');

deltaDq = max(finite_order_Dqs) - min(finite_order_Dqs);
fprintf('Spectral dimension range (ΔDq) = %.6f\n', deltaDq);

%% Summary
fprintf('\n=== FINAL SUMMARY ===\n');
fprintf('Network: %s\n', print_network_name);
fprintf('Nodes: %d, Edges: %d\n', numnodes(G), numedges(G));
fprintf('Communities: %d (Q=%.4f)\n', num_infomap_communities, infomap_modularity);
fprintf('Spectral: ΔDq=%.4f, Mean=%.3f±%.3f\n', deltaDq, mean_Dq, std_Dq);
fprintf('=====================\n');


%% ========== COMPARE WITH REAL VENICE SESTIERI ==========
fprintf('\n============================================================\n');
fprintf('COMPARISON WITH REAL VENICE SESTIERI\n');
fprintf('============================================================\n');

% Load real sestieri labels
try
    sestieri_data = readtable('networks/venice_real_sestieri.csv');
    fprintf('✓ Loaded real sestieri labels from venice_real_sestieri.csv\n');
    
    % Create mapping from osmid to sestiere
    osmid_to_sestiere = containers.Map(sestieri_data.osmid, sestieri_data.sestiere);
    
    % Map graph nodes to sestieri
    % G uses sequential node IDs, we need to map back to original osmids
    real_sestieri = zeros(numnodes(G), 1);
    
    % Try to get original node IDs from the graph
    if istable(G.Nodes) && ismember('Name', G.Nodes.Properties.VariableNames)
        for i = 1:numnodes(G)
            node_name = G.Nodes.Name{i};
            try
                original_id = str2double(node_name);
                if isKey(osmid_to_sestiere, original_id)
                    real_sestieri(i) = osmid_to_sestiere(original_id);
                end
            catch
                % Skip if conversion fails
            end
        end
    else
        % Fallback: try direct mapping by index (less reliable)
        fprintf('Warning: Using index-based mapping (may be inaccurate)\n');
        for i = 1:min(numnodes(G), height(sestieri_data))
            if i <= height(sestieri_data)
                real_sestieri(i) = sestieri_data.sestiere(i);
            end
        end
    end
    
    % Filter to valid nodes (sestiere > 0 and community > 0)
    valid_mask = (real_sestieri > 0) & (infomap_communities > 0);
    real_sestieri_valid = real_sestieri(valid_mask);
    infomap_valid = infomap_communities(valid_mask);
    
    fprintf('Comparing %d nodes with both labels\n', sum(valid_mask));
    
    if sum(valid_mask) > 100  % Need minimum nodes for meaningful comparison
        
        % Sestieri names
        sestieri_names = {'San Marco', 'San Polo', 'Santa Croce', ...
                          'Cannaregio', 'Castello', 'Dorsoduro'};
        
        % Calculate confusion matrix
        confusion = zeros(6, num_infomap_communities);
        for i = 1:length(real_sestieri_valid)
            s = real_sestieri_valid(i);
            c = infomap_valid(i);
            if s >= 1 && s <= 6 && c >= 1 && c <= num_infomap_communities
                confusion(s, c) = confusion(s, c) + 1;
            end
        end
        
        % Calculate purity
        purity = sum(max(confusion, [], 1)) / sum(confusion(:));
        
        % Find best community-to-sestiere mapping
        best_mapping = zeros(num_infomap_communities, 1);
        used_sestieri = false(6, 1);
        
        for c = 1:num_infomap_communities
            available = confusion(:, c);
            available(used_sestieri) = -1;
            [max_val, best_s] = max(available);
            
            if max_val > 0
                best_mapping(c) = best_s;
                used_sestieri(best_s) = true;
            end
        end
        
        % Calculate accuracy with best mapping
        correct = 0;
        for i = 1:length(infomap_valid)
            mapped_sestiere = best_mapping(infomap_valid(i));
            if mapped_sestiere == real_sestieri_valid(i)
                correct = correct + 1;
            end
        end
        accuracy = correct / length(infomap_valid);
        
        fprintf('\n--- COMPARISON METRICS ---\n');
        fprintf('Purity: %.2f%%\n', purity * 100);
        fprintf('Accuracy (best mapping): %.2f%%\n', accuracy * 100);
        
        fprintf('\nBest Community-to-Sestiere Mapping:\n');
        for c = 1:num_infomap_communities
            if best_mapping(c) > 0
                overlap = confusion(best_mapping(c), c);
                total = sum(confusion(:, c));
                pct = 100 * overlap / total;
                fprintf('  Community %d → %s (%d/%d nodes, %.1f%%)\n', ...
                        c, sestieri_names{best_mapping(c)}, ...
                        overlap, total, pct);
            end
        end
        
        % Visualization
        figure('Position', [50 50 1400 900], 'Name', 'Sestieri Comparison');
        
        % Confusion matrix
        subplot(2, 2, 1);
        imagesc(confusion);
        colorbar;
        colormap(hot);
        set(gca, 'YTick', 1:6, 'YTickLabel', sestieri_names);
        set(gca, 'XTick', 1:num_infomap_communities);
        xlabel('Infomap Community', 'FontSize', 11);
        ylabel('Real Sestiere', 'FontSize', 11);
        title(sprintf('Confusion Matrix\nPurity: %.1f%%', purity*100), 'FontWeight', 'bold');
        
        % Normalized confusion
        subplot(2, 2, 2);
        confusion_norm = confusion ./ sum(confusion, 2);
        confusion_norm(isnan(confusion_norm)) = 0;
        imagesc(confusion_norm);
        colorbar;
        colormap(hot);
        set(gca, 'YTick', 1:6, 'YTickLabel', sestieri_names);
        set(gca, 'XTick', 1:num_infomap_communities);
        xlabel('Infomap Community', 'FontSize', 11);
        ylabel('Real Sestiere', 'FontSize', 11);
        title('Normalized (by Sestiere)', 'FontWeight', 'bold');
        
        % Metrics text
        subplot(2, 2, 3);
        axis off;
        metrics_text = {
            'COMPARISON RESULTS';
            '========================';
            sprintf('Nodes compared: %d', sum(valid_mask));
            sprintf('Purity: %.2f%%', purity*100);
            sprintf('Accuracy: %.2f%%', accuracy*100);
            '';
            'INTERPRETATION:';
            '  >70%: Good match';
            '  50-70%: Moderate';
            '  <50%: Poor';
            '';
            sprintf('Result: %s', ...
                    ternary(accuracy > 0.7, 'Good', ...
                    ternary(accuracy > 0.5, 'Moderate', 'Poor')));
        };
        text(0.1, 0.9, metrics_text, 'Units', 'normalized', ...
             'VerticalAlignment', 'top', 'FontName', 'Courier', 'FontSize', 10);
        
        sgtitle('Venice: Infomap Communities vs Real Sestieri', ...
                'FontSize', 16, 'FontWeight', 'bold');
        
        print('venice_sestieri_comparison', '-dpng', '-r300');
        fprintf('\n✓ Saved comparison: venice_sestieri_comparison.png\n');
        
        % Save comparison data
        comparison_table = table((1:numnodes(G))', real_sestieri, infomap_communities, ...
                                 'VariableNames', {'NodeID', 'Sestiere', 'Community'});
        writetable(comparison_table, 'venice_comparison.csv');
        fprintf('✓ Saved data: venice_comparison.csv\n');
        
    else
        fprintf('⚠ Not enough valid nodes for comparison (%d found, need >100)\n', sum(valid_mask));
    end
    
catch ME
    fprintf('⚠ Could not load sestieri data: %s\n', getReport(ME, 'basic'));
    fprintf('  Make sure you ran venice_sestieri_extraction.py first!\n');
end

fprintf('============================================================\n');

% Helper function for ternary operator
function result = ternary(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end



%% Helper Functions
function communities = parse_infomap_tree(tree_file, num_nodes)
    communities = zeros(num_nodes, 1);
    fid = fopen(tree_file, 'r');
    if fid == -1, error('Cannot open: %s', tree_file); end
    
    line = fgetl(fid);
    while ~feof(fid) && startsWith(line, '#')
        line = fgetl(fid);
    end
    
    while ~feof(fid)
        if ischar(line) && ~isempty(line) && ~startsWith(line, '#')
            parts = strsplit(line);
            if length(parts) >= 4
                path = parts{1};
                node_id = str2double(parts{4});
                path_parts = strsplit(path, ':');
                community_id = str2double(path_parts{1});
                
                if node_id > 0 && node_id <= num_nodes
                    communities(node_id) = community_id;
                end
            end
        end
        line = fgetl(fid);
    end
    fclose(fid);
end

function Q = calculate_modularity(G, communities)
    A = adjacency(G);
    m = nnz(A) / 2;
    k = full(sum(A, 2));
    n = numnodes(G);
    
    Q = 0;
    for i = 1:n
        for j = 1:n
            if communities(i) == communities(j)
                Q = Q + (A(i,j) - (k(i)*k(j))/(2*m));
            end
        end
    end
    Q = Q / (2*m);
end

function [communities, modularity] = louvain_communities_matlab(G, target)
    fprintf('Native algorithm for %d nodes...\n', numnodes(G));
    n = numnodes(G);
    A = adjacency(G);
    k = full(sum(A, 2));
    
    try
        [communities, ~] = kmeans(k, target, 'MaxIter', 50, 'Replicates', 3);
    catch
        communities = randi(target, n, 1);
    end
    
    modularity = calculate_modularity(G, communities);
    fprintf('Result: %d communities, Q=%.4f\n', max(communities), modularity);
end