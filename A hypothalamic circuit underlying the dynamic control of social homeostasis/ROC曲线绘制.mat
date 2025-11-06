%% Generate Publication-Quality Single-Neuron ROC Plots (Standalone Script)
% This is a self-contained script. It takes a single raw data file,
% performs the full ROC analysis internally, finds the best example neurons,
% and generates a detailed ROC analysis figure.
% No pre-processing or other scripts are needed.

clear; clc; close all;

%% =================== 1. 用户设置 =======================================
% --
file_to_analyze = "F:\工作文件\RA\数据集\Fig2C\FVB_Panneuronal_Reunion_Isolation_Day3_Mouse#3_preprocessed.mat";

% --- (可选) 手动指定神经元ID ---
% 如果您想手动指定神经元，请取消下面的注释并填入ID。
% 否则，脚本会自动寻找AUC最高和最低的神经元。
% manual_neuron_id_excited = 112;
% manual_neuron_id_inhibited = 45;


%% =================== 2. 数据加载和预处理 ==============================
[filepath, basename, ~] = fileparts(file_to_analyze);
disp("加载并处理文件: " + basename);

% 加载原始数据
load(file_to_analyze); % 加载 results_final

% 定义时间窗口和参数
reunionStart = 838; % 确保这个值与您期望的事件时间一致
baselineDuration = 300;
reunionDuration = 300;
baselineStart = reunionStart - baselineDuration; 
baselineEnd = reunionStart - 1;
reunionEnd = reunionStart + reunionDuration - 1;
response_bins = 300;
ROC_step = 0.01;

% 分箱和Z-score标准化
raw_data = results_final.C_raw_final;
nNeurons = size(raw_data, 1);
num_bins = floor(size(raw_data, 2)/10);
bin_data = zeros(nNeurons, num_bins);
for m = 1:nNeurons
    for p = 1:num_bins
        bin_data(m,p) = mean(raw_data(m, ((p*10)-9):(p*10)));
    end
end
zscored_data = zscore(bin_data,[],2);

%% =================== 3. 内部ROC分析 ===================================
% 对文件中的所有神经元进行ROC分析，以找到最佳示例
disp('正在对所有神经元进行ROC分析以寻找最佳示例...');
AUC_mat = zeros(nNeurons, 2); % 只需要存储ID和AUC值
for n = 1:nNeurons
    if mod(n, 50) == 0; disp("  ... " + n + "/" + nNeurons); end
    
    temp_neuron = zscored_data(n,:);
    temp_baseline = temp_neuron(baselineStart:baselineEnd);
    temp_reunion = temp_neuron(reunionStart:reunionEnd);
        
    % 只计算AUC值，不需要显著性检验
    min_value = min([temp_baseline temp_reunion]);
    max_value = max([temp_baseline temp_reunion]);
    thresholds = min_value:ROC_step:max_value;
    p_Hit = zeros(1, length(thresholds));
    p_FA = zeros(1, length(thresholds));
    for k = 1:length(thresholds)
        thresh = thresholds(k);
        p_Hit(k) = sum(temp_reunion >= thresh) / response_bins;
        p_FA(k) = sum(temp_baseline >= thresh) / response_bins;
    end
    
    AUC_mat(n,1) = n;
    AUC_mat(n,2) = -trapz(p_FA, p_Hit);
end
disp('...分析完成！');

%% =================== 4. 寻找或确认示例神经元 ==========================
if exist('manual_neuron_id_inhibited', 'var')
    neuron_id_inhibited = manual_neuron_id_inhibited;
    disp(['使用手动指定的抑制性神经元: ID ' num2str(neuron_id_inhibited)]);
else
    % 自动寻找AUC最低的神经元
    [min_auc, idx] = min(AUC_mat(:,2));
    neuron_id_inhibited = AUC_mat(idx, 1);
    disp(['自动找到最强抑制性神经元: ID ' num2str(neuron_id_inhibited) ' (AUC = ' num2str(min_auc) ')']);
end

if exist('manual_neuron_id_excited', 'var')
    neuron_id_excited = manual_neuron_id_excited;
    disp(['使用手动指定的兴奋性神经元: ID ' num2str(neuron_id_excited)]);
else
    % 自动寻找AUC最高的神经元
    [max_auc, idx] = max(AUC_mat(:,2));
    neuron_id_excited = AUC_mat(idx, 1);
    disp(['自动找到最强兴奋性神经元: ID ' num2str(neuron_id_excited) ' (AUC = ' num2str(max_auc) ')']);
end

%% =================== 5. 生成组合图 ====================================
disp('正在生成图片...');

fig = figure('Position', [100, 100, 1000, 450], 'Color', 'w');

% --- 绘制左侧面板 (抑制性神经元) ---
neuron_trace_I = zscored_data(neuron_id_inhibited, :);
baseline_data_I = neuron_trace_I(baselineStart:baselineEnd);
test_data_I = neuron_trace_I(reunionStart:reunionEnd);

ax_hist_I = subplot(1, 4, 1);
ax_roc_I = subplot(1, 4, 2);
color_I = [0.47 0.67 0.19]; % 从示例图中提取的绿色
title_text_I = "Example MPN^{Isolation} neuron";
plot_single_neuron_roc(ax_hist_I, ax_roc_I, baseline_data_I, test_data_I, color_I, title_text_I, 'a');

% --- 绘制右侧面板 (兴奋性神经元) ---
neuron_trace_E = zscored_data(neuron_id_excited, :);
baseline_data_E = neuron_trace_E(baselineStart:baselineEnd);
test_data_E = neuron_trace_E(reunionStart:reunionEnd);

ax_hist_E = subplot(1, 4, 3);
ax_roc_E = subplot(1, 4, 4);
color_E = [0.58 0.40 0.74]; % 从示例图中提取的紫色
title_text_E = "Example MPN^{Reunion} neuron";
plot_single_neuron_roc(ax_hist_E, ax_roc_E, baseline_data_E, test_data_E, color_E, title_text_E, 'b');

% 保存最终图片
output_filename = fullfile(filepath, [basename '_Standalone_Example_ROC_Plots.png']);
saveas(fig, output_filename);
disp(['图片已保存为: ' output_filename]);


%% =================== 辅助函数：用于绘制单个面板 ========================
function plot_single_neuron_roc(ax_hist, ax_roc, baseline_data, test_data, plot_color, title_text, panel_label)
    
    % --- 1. 绘制直方图 ---
    axes(ax_hist);
    hold on;
    
    % 确定一个统一的bin边，以便两个直方图可以完美对齐
    combined_data = [baseline_data, test_data];
    bin_edges = linspace(min(combined_data), max(combined_data), 15);

    % 绘制基线期分布 (灰色)
    h1 = histogram(baseline_data, bin_edges, 'Normalization', 'probability', ...
        'FaceColor', [0.5 0.5 0.5], 'EdgeColor', 'none');
        
    % 绘制反应期分布 (彩色)
    h2 = histogram(test_data, bin_edges, 'Normalization', 'probability', ...
        'FaceColor', plot_color, 'EdgeColor', 'none');
    
    % 美化直方图
    title(title_text, 'Color', plot_color, 'FontWeight', 'normal', 'FontSize', 12);
    xlabel('Activity amplitude (Z-score)');
    ylabel('Probability');
    legend([h2, h1], {'Reunion', 'Isolation (baseline)'}, 'Location', 'northwest', 'Box', 'off');
    set(gca, 'FontSize', 10, 'FontName', 'Arial', 'TickDir', 'out');
    box off;
    
    text(-0.25, 1.1, panel_label, 'Units', 'normalized', 'FontSize', 16, 'FontWeight', 'bold');

    % --- 2. 绘制ROC曲线 ---
    axes(ax_roc);
    hold on;
    
    min_value = min(combined_data);
    max_value = max(combined_data);
    thresholds = linspace(min_value, max_value, 200);
    
    p_Hit = zeros(1, length(thresholds));
    p_FA = zeros(1, length(thresholds));
    
    for k = 1:length(thresholds)
        thresh = thresholds(k);
        p_Hit(k) = sum(test_data >= thresh) / length(test_data);
        p_FA(k) = sum(baseline_data >= thresh) / length(baseline_data);
    end
    
    AUC = -trapz(p_FA, p_Hit);
    
    plot(p_FA, p_Hit, 'Color', plot_color, 'LineWidth', 2);
    plot([0 1], [0 1], 'k--', 'LineWidth', 1);
    
    xlabel('False positive rate');
    ylabel('True positive rate');
    axis square;
    xlim([0 1]);
    ylim([0 1]);
    set(gca, 'FontSize', 10, 'FontName', 'Arial', 'TickDir', 'out');
    
    text(0.5, 0.8, ['AUC=' num2str(AUC, '%.2f')], 'HorizontalAlignment', 'center', 'FontSize', 12);
    box off;
end
