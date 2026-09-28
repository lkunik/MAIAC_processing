
input_filename = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/MCD19_Fluxsite_99_20210322.mat';
data = load(input_filename);

% Set the output directory
outputDir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/MODIS/MAIAC_MCD19A1/from_Ran_by_fluxsite/';


% Display what variables were loaded
disp('Variables in the dataset:');
disp(fieldnames(data));

% Look at the T field (likely the main data table)
disp('T field info:');
disp(size(data.T));
disp(class(data.T));


% If T is a table, show column names
if istable(data.T)
    disp('Column names in T:');
    disp(data.T.Properties.VariableNames);
end


% Look at datasite field
disp('Datasite field:');
disp(data.datasite);
disp(size(data.datasite));

% Look at meta field
disp('Meta field:');
disp(data.meta);

% Look at siteInfo2018 field
disp('SiteInfo2018 field:');
disp(data.siteInfo2018);

% Get the site information
siteInfo = data.siteInfo2018;
mainTable = data.T;

% Get unique site indices
uniqueSiteIndices = unique(mainTable.siteIndex);
disp(['Found ', num2str(length(uniqueSiteIndices)), ' unique sites']);

% Loop through each site
for i = 1:length(uniqueSiteIndices)
    currentSiteIndex = uniqueSiteIndices(i);
    
    % Find the site ID for this index
    siteRow = find(siteInfo.siteIndex == currentSiteIndex);
    if ~isempty(siteRow)
        siteID = siteInfo.siteID{siteRow};
        disp(['Processing site ', num2str(i), ': ', siteID, ' (index: ', num2str(currentSiteIndex), ')']);
    else
        siteID = ['Site_' num2str(currentSiteIndex)];
        disp(['Processing site ', num2str(i), ': Unknown site (index: ', num2str(currentSiteIndex), ')']);
    end
    
    % Extract data for this site
    siteData = mainTable(mainTable.siteIndex == currentSiteIndex, :);
    disp(['  Found ', num2str(height(siteData)), ' records for this site']);
    
    % Create filename with full path
    cleanSiteID = strrep(siteID, '-', '_');
    filename = fullfile(outputDir, [cleanSiteID, '.csv']);
    
    % Write to CSV
    writetable(siteData, filename);
    disp(['  Saved to: ', filename]);
end

disp('All sites processed!');