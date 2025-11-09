%% ========================================================================
%  整合版ROC分析与发表级图片生成脚本 - V6.1
%  - 堆叠图更新：使用方括号明确关联曲线组与各自的比例尺
% ========================================================================

clear; clc; close all;

%% ========================================================================
%  1. 用户配置区域
% ========================================================================

base_folder = 'F:\工作文件\RA\数据集\Fig2C';
file_names = {
    'FVB_Panneuronal_Reunion_Isolation_Day3_Mouse#1_preprocessed.mat'};
file_list = fullfile(base_folder, file_names);
output_folder = fullfile(base_folder, 'Analysis_Output_Batch');
reunionStart      = 911;
%%：注意！！！reunionStart每个小鼠的不一样，必须先运行时间点检测的代码！！！
baselineDuration  = 300;
reunionDuration   = 300;
ROC_step          = 0.01;
AUC_threshold_E   = 0.65;
AUC_threshold_I   = 0.35;

%% ========================================================================
%  2. 初始化与预检查 (无变化)
% ========================================================================
disp('--- 开始批量分析流程 ---');
if ~exist(output_folder, 'dir'), mkdir(output_folder); disp(['已创建输出文件夹: ', output_folder]); end
baselineStart = reunionStart - baselineDuration; baselineEnd = reunionStart - 1; reunionEnd = reunionStart + reunionDuration - 1; response_bins = baselineDuration;
min_length = inf;
disp('预检查文件时长...');
for i = 1:length(file_list)
    try, s = load(file_list{i}, 'results_final'); num_bins = floor(size(s.results_final.C_raw_final, 2) / 10); min_length = min(min_length, num_bins);
    catch ME, warning(['加载文件失败: ', file_list{i}, ' | ', ME.message]); end
end
disp(['记录将被统一裁剪至: ', num2str(min_length), ' 秒。']);
if isinf(min_length), error('所有文件均加载失败，无法继续分析。请检查base_folder路径是否正确！'); end
if min_length < reunionEnd, error('记录时长不足以覆盖分析窗口。'); end
agg_zscored_data = {}; agg_AUC_mat = {}; agg_mouse_names = {};

%% ========================================================================
%  3. 主循环：依次处理每个文件 (无变化)
% ========================================================================
for i = 1:length(file_list)
    current_file_path = file_list{i};
    [~, basename, ~] = fileparts(current_file_path);
    clean_basename = strtrim(basename); clean_basename = strrep(clean_basename, '#', 'Num'); clean_basename = strrep(clean_basename, '(', ''); clean_basename = strrep(clean_basename, ')', '');
    disp('================================================================'); disp(['开始处理文件 (', num2str(i), '/', num2str(length(file_list)), '): ', basename]); disp('================================================================');
    try, load(current_file_path, 'results_final'); raw_data = results_final.C_raw_final;
    catch ME, warning(['加载失败，跳过: ', basename, ' | ', ME.message]); continue; end
    nNeurons = size(raw_data, 1); num_bins = floor(size(raw_data, 2) / 10); bin_data = zeros(nNeurons, num_bins);
    for m = 1:nNeurons, for p = 1:num_bins, bin_data(m,p) = mean(raw_data(m, ((p*10)-9):(p*10))); end, end
    bin_data = bin_data(:, 1:min_length); zscored_data = zscore(bin_data, [], 2);
    h_raw_heatmap = figure('Visible', 'off');
    imagesc(zscored_data); caxis([-1 5]); colormap('jet');
    line([baselineStart,baselineStart], [0,nNeurons+0.5], 'Color', 'w'); line([reunionStart,reunionStart], [0,nNeurons+0.5], 'Color', 'w'); line([reunionEnd,reunionEnd], [0,nNeurons+0.5], 'Color', 'w');
    title([basename, ' Raw Heatmap'], 'Interpreter', 'none');
    heatmap_filename = fullfile(output_folder, [clean_basename, '_raw_heatmap.jpg']);
    saveas(h_raw_heatmap, heatmap_filename);
    close(h_raw_heatmap);
    disp("  ...ROC分析..."); AUC_mat = zeros(nNeurons, 4);
    for n = 1:nNeurons
        temp_neuron = zscored_data(n,:);
        [AUC, sig, sig70] = DL_AUC(temp_neuron(baselineStart:baselineEnd), temp_neuron(reunionStart:reunionEnd), ROC_step, response_bins, AUC_threshold_E, AUC_threshold_I);
        AUC_mat(n,1:4) = [n, AUC, sig, sig70];
    end
    disp("  ...ROC分析完成。");
    sig70_E_reu_mat = AUC_mat(AUC_mat(:,4)==1,:); sig70_I_reu_mat = AUC_mat(AUC_mat(:,4)==-1,:);
    results_filename = fullfile(output_folder, [clean_basename, '_analysis_results.mat']);
    save(results_filename, 'AUC_mat', 'sig70_E_reu_mat', 'sig70_I_reu_mat');
    disp("  -> 单个文件分析完成！结果已保存。");
    agg_zscored_data{end+1} = zscored_data; agg_AUC_mat{end+1} = AUC_mat; agg_mouse_names{end+1} = clean_basename;
end
disp('================================================================'); disp('所有文件的独立分析均已完成！'); disp('================================================================');

%% ========================================================================
%  4. 聚合所有数据 (无变化)
% ========================================================================
disp('--- 开始聚合所有小鼠数据以生成最终图片 ---');
final_zscored_data = vertcat(agg_zscored_data{:});
final_AUC_mat = []; neuron_offset = 0;
for i = 1:length(agg_AUC_mat)
    temp_AUC_mat = agg_AUC_mat{i}; 
    temp_AUC_mat(:,1) = temp_AUC_mat(:,1) + neuron_offset;
    final_AUC_mat = [final_AUC_mat; temp_AUC_mat];
    neuron_offset = neuron_offset + size(agg_AUC_mat{i}, 1);
end
disp('...数据聚合完毕！');

%% ========================================================================
%  5. 生成最终发表级聚合图 (PUBLICATION_FIGURE) (无变化)
% ========================================================================
disp('正在生成最终发表级聚合图...');
sig70_I_all = final_AUC_mat(final_AUC_mat(:,4)==-1,:); sig70_E_all = final_AUC_mat(final_AUC_mat(:,4)==1,:);
sorted_I = sortrows(sig70_I_all, 2, 'ascend'); sorted_E = sortrows(sig70_E_all, 2, 'descend');
if ~isempty(sorted_I), data_I = final_zscored_data(sorted_I(:,1),:); else, data_I = []; end
if ~isempty(sorted_E), data_E = final_zscored_data(sorted_E(:,1),:); else, data_E = []; end
total_neurons = size(final_zscored_data, 1); time_axis = (1:min_length) - reunionStart;
fig = figure('Position', [50, 50, 900, 700], 'Color', 'w');
ax1 = subplot(2, 2, 1); if ~isempty(data_I), imagesc(time_axis, 1:size(data_I,1), data_I); end; colormap(ax1, 'jet'); caxis([-1, 3]); hold on;
line([0, 0], ylim, 'Color', 'w', 'LineWidth', 1.5); line([reunionDuration, reunionDuration], ylim, 'Color', 'w', 'LineStyle', '--', 'LineWidth', 1.5);
title('MPN^{Isolation} neurons', 'Color', [0.2 0.7 0.3], 'FontSize', 14, 'FontWeight', 'normal'); ylabel('Neurons'); set(gca, 'XTickLabel', [], 'FontSize', 10, 'FontName', 'Arial');
ax2 = subplot(2, 2, 2); if ~isempty(data_E), imagesc(time_axis, 1:size(data_E,1), data_E); end; colormap(ax2, 'jet'); caxis([-1, 3]); hold on;
line([0, 0], ylim, 'Color', 'w', 'LineWidth', 1.5); line([reunionDuration, reunionDuration], ylim, 'Color', 'w', 'LineStyle', '--', 'LineWidth', 1.5);
title('MPN^{Reunion} neurons', 'Color', [0.6 0.2 0.8], 'FontSize', 14, 'FontWeight', 'normal'); set(gca, 'XTickLabel', [], 'YTickLabel', [], 'FontSize', 10, 'FontName', 'Arial');
cb = colorbar(ax2, 'Position', [0.92 0.7 0.02 0.15]); ylabel(cb, 'Activity (Z score)');
ax3 = subplot(2, 2, 3); if ~isempty(data_I), mean_I = mean(data_I, 1); sem_I = std(data_I, 0, 1) / sqrt(size(data_I, 1));
fill([time_axis, fliplr(time_axis)], [mean_I-sem_I, fliplr(mean_I+sem_I)], [0.2 0.7 0.3], 'FaceAlpha', 0.2, 'EdgeColor', 'none'); hold on;
plot(time_axis, mean_I, 'Color', [0.2 0.7 0.3], 'LineWidth', 1.5); end; line([0, 0], [-0.5, 1], 'Color', 'k'); line([reunionDuration, reunionDuration], [-0.5, 1], 'Color', 'k', 'LineStyle', '--');
ylim([-0.5, 1]); xlim([time_axis(1), time_axis(end)]); xlabel('Time from reunion (s)', 'FontName', 'Arial', 'FontSize', 12); box off; axis off;
if ~isempty(data_I), text(-280, 0.8, [num2str(size(data_I,1)), '/', num2str(total_neurons)], 'FontSize', 12, 'FontName', 'Arial'); text(-280, 0.6, 'neurons', 'FontSize', 12, 'FontName', 'Arial'); end
ax4 = subplot(2, 2, 4); if ~isempty(data_E), mean_E = mean(data_E, 1); sem_E = std(data_E, 0, 1) / sqrt(size(data_E, 1));
fill([time_axis, fliplr(time_axis)], [mean_E-sem_E, fliplr(mean_E+sem_E)], [0.6 0.2 0.8], 'FaceAlpha', 0.2, 'EdgeColor', 'none'); hold on;
plot(time_axis, mean_E, 'Color', [0.6 0.2 0.8], 'LineWidth', 1.5); end; line([0, 0], [-0.5, 2.5], 'Color', 'k'); line([reunionDuration, reunionDuration], [-0.5, 2.5], 'Color', 'k', 'LineStyle', '--');
ylim([-0.5, 2.5]); xlim([time_axis(1), time_axis(end)]); xlabel('Time from reunion (s)', 'FontName', 'Arial', 'FontSize', 12); box off; axis off;
if ~isempty(data_E), text(-280, 2.2, [num2str(size(data_E,1)), '/', num2str(total_neurons)], 'FontSize', 12, 'FontName', 'Arial'); text(-280, 1.9, 'neurons', 'FontSize', 12, 'FontName', 'Arial'); end
linkaxes([ax1, ax2, ax3, ax4], 'x');
final_figure_path = fullfile(output_folder, 'PUBLICATION_FIGURE.png');
saveas(fig, final_figure_path);
disp('...最终发表级聚合图生成完毕！');

%% ========================================================================
%  6. 生成堆叠式代表性神经元活动曲线图 (抑制性置顶 + 括号比例尺)
% ========================================================================
disp('正在生成堆叠式代表性神经元活动曲线图...');
sig70_I_all = final_AUC_mat(final_AUC_mat(:,4)==-1, :);
sig70_E_all = final_AUC_mat(final_AUC_mat(:,4)==1, :);
sig_NS_all = final_AUC_mat(final_AUC_mat(:,3)==0, :);
sorted_I_neurons = sortrows(sig70_I_all, 2, 'ascend');
sorted_E_neurons = sortrows(sig70_E_all, 2, 'descend');
dist_to_0_5 = abs(sig_NS_all(:, 2) - 0.5);
[~, sorted_indices] = sort(dist_to_0_5, 'ascend');
sorted_NS_neurons = sig_NS_all(sorted_indices, :);
num_E_to_plot = min(4, size(sorted_E_neurons, 1));
num_I_to_plot = min(4, size(sorted_I_neurons, 1));
num_NS_to_plot = min(2, size(sorted_NS_neurons, 1));
selected_E_neurons = sorted_E_neurons(1:num_E_to_plot, :);
selected_I_neurons = sorted_I_neurons(1:num_I_to_plot, :);
selected_NS_neurons = sorted_NS_neurons(1:num_NS_to_plot, :);
all_selected_neurons = [selected_I_neurons; selected_E_neurons; selected_NS_neurons];
all_selected_ids = all_selected_neurons(:, 1);
window_indices = (reunionStart - 300) : (reunionStart + 300);
all_selected_traces = final_zscored_data(all_selected_ids, window_indices);
window_time_axis = -300:300;
stacked_traces_path_base = fullfile(output_folder, 'STACKED_TRACES_FIGURE');
plot_stacked_traces(all_selected_traces, all_selected_neurons, window_time_axis, reunionDuration, stacked_traces_path_base);
disp('...堆叠式曲线图生成完毕！');
disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'); disp(['!!!         全部分析流程顺利完成！                     !!!']); disp(['!!!  所有结果均保存在: ', output_folder, '  !!!']); disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!');

%% ========================================================================
%  函数定义
% ========================================================================
function [AUC, sig, sig70] = DL_AUC(baseline, test, ROC_step, response_bins, threshold_E, threshold_I)
    baseline = baseline(:)'; test = test(:)'; min_value = min([baseline test]); max_value = max([baseline test]);
    if min_value == max_value, AUC = 0.5; sig = 0; sig70 = 0; return; end
    thresholds = min_value:ROC_step:max_value; p_Hit = zeros(1, length(thresholds)); p_FA = zeros(1, length(thresholds));
    for k = 1:length(thresholds), thresh = thresholds(k); p_Hit(k) = sum(test >= thresh) / response_bins; p_FA(k) = sum(baseline >= thresh) / response_bins; end
    AUC = -trapz(p_FA, p_Hit); orig_data = [baseline test]; shuffled_AUC = zeros(1, 1000);
    for s=1:1000
        shuffled_ID = randperm(response_bins*2); shuff_baseline = orig_data(shuffled_ID(1:response_bins)); shuff_resp = orig_data(shuffled_ID(response_bins+1:end));
        shuff_p_Hit = zeros(1, length(thresholds)); shuff_p_FA = zeros(1, length(thresholds));
        for shuff_k = 1:length(thresholds), shuff_thresh = thresholds(shuff_k); shuff_p_Hit(shuff_k) = sum(shuff_resp >= shuff_thresh) / response_bins; shuff_p_FA(shuff_k) = sum(shuff_baseline >= shuff_thresh) / response_bins; end
        shuffled_AUC(s) = -trapz(shuff_p_FA, shuff_p_Hit);
    end
    Active_Cutoff = prctile(shuffled_AUC, 97.5); Inhib_Cutoff = prctile(shuffled_AUC, 2.5);
    if AUC > Active_Cutoff, sig = 1; elseif AUC < Inhib_Cutoff, sig = -1; else, sig = 0; end
    if sig == 1 && AUC >= threshold_E, sig70 = 1; elseif sig == -1 && AUC <= threshold_I, sig70 = -1; else, sig70 = 0; end
end

% ========================================================================
%  【核心修改】用于生成堆叠式曲线图的函数 (括号比例尺)
% ========================================================================
function plot_stacked_traces(traces_data, neurons_info, time_axis, reunion_duration, output_filename_base)
    num_traces = size(traces_data, 1);
    if num_traces == 0, disp('没有可供绘制的代表性神经元。'); return; end
    vertical_spacing = 8; 
    line_width = 1.5;
    color_excited = [0.8500, 0.3250, 0.0980];
    color_inhibited = [0, 0.4470, 0.7410];
    color_ns = [0.5, 0.5, 0.5];
    inhib_scaling_factor = 2.0;
    
    fig = figure('Position', [100, 100, 650, 800], 'Color', 'w');
    hold on;
    y_ticks = zeros(1, num_traces);
    y_labels = cell(1, num_traces + 1);
    y_labels{1} = '#';
    
    for k = 1:num_traces
        offset = (num_traces - k) * vertical_spacing;
        neuron_type = neurons_info(k, 4);
        scaling_factor = 1.0;
        if neuron_type == -1
            scaling_factor = inhib_scaling_factor;
            current_color = color_inhibited;
        elseif neuron_type == 1
            current_color = color_excited;
        else
            current_color = color_ns;
        end
        plot(time_axis, traces_data(k, :) * scaling_factor + offset, 'Color', current_color, 'LineWidth', line_width);
        y_ticks(k) = offset;
        neuron_id = neurons_info(k, 1);
        if neuron_type == 0
            y_labels{k+1} = 'n.s.';
        else
            y_labels{k+1} = num2str(neuron_id);
        end
    end
    
    ax = gca;
    ax.YTick = fliplr(y_ticks);
    ax.YTickLabel = y_labels;
    ax.TickDir = 'out';
    ax.YAxis.FontSize = 12;
    ax.XAxis.Visible = 'off';
    ylim([-vertical_spacing, num_traces * vertical_spacing]);
    box off;
    
    plot_ylim = ylim;
    line([0, 0], plot_ylim, 'Color', 'k', 'LineWidth', 1);
    line([reunion_duration, reunion_duration], plot_ylim, 'Color', 'k', 'LineWidth', 1);
    text(0, plot_ylim(2), 'Partner in', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 12);
    text(reunion_duration, plot_ylim(2), 'Partner out', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize',12);

    % --- 【核心修改】绘制带括号的比例尺系统 ---
    bracket_x = time_axis(end) + 20;
    bracket_tick_len = 10;
    
    % 1. 抑制性神经元组
    num_I = sum(neurons_info(:,4) == -1);
    if num_I > 0
        y_top_I = (num_traces - 0.5) * vertical_spacing;
        y_bottom_I = (num_traces - num_I + 0.5) * vertical_spacing;
        % 绘制垂直括号线
        plot([bracket_x, bracket_x], [y_bottom_I, y_top_I], 'k-');
        % 绘制上下短横线
        plot([bracket_x, bracket_x - bracket_tick_len], [y_top_I, y_top_I], 'k-');
        plot([bracket_x, bracket_x - bracket_tick_len], [y_bottom_I, y_bottom_I], 'k-');
        
        % 绘制专属比例尺
        scale_bar_x = bracket_x + 15;
        scale_bar_y_center = (y_top_I + y_bottom_I) / 2;
        scale_bar_height = 5 * inhib_scaling_factor;
        plot([scale_bar_x, scale_bar_x], [scale_bar_y_center - scale_bar_height/2, scale_bar_y_center + scale_bar_height/2], 'k', 'LineWidth', 2);
        text(scale_bar_x + 5, scale_bar_y_center, '5\sigma', 'FontSize', 12, 'HorizontalAlignment', 'left');
    end
    
    % 2. 兴奋性/非显著神经元组
    num_E_NS = num_traces - num_I;
    if num_E_NS > 0
        y_top_E_NS = (num_E_NS - 0.5) * vertical_spacing;
        y_bottom_E_NS = -0.5 * vertical_spacing;
        % 绘制垂直括号线
        plot([bracket_x, bracket_x], [y_bottom_E_NS, y_top_E_NS], 'k-');
        % 绘制上下短横线
        plot([bracket_x, bracket_x - bracket_tick_len], [y_top_E_NS, y_top_E_NS], 'k-');
        plot([bracket_x, bracket_x - bracket_tick_len], [y_bottom_E_NS, y_bottom_E_NS], 'k-');
        
        % 绘制标准比例尺
        scale_bar_x = bracket_x + 15;
        scale_bar_y_center = (y_top_E_NS + y_bottom_E_NS) / 2;
        scale_bar_height = 5; % 标准 5 sigma
        plot([scale_bar_x, scale_bar_x], [scale_bar_y_center - scale_bar_height/2, scale_bar_y_center + scale_bar_height/2], 'k', 'LineWidth', 2);
        text(scale_bar_x + 5, scale_bar_y_center, '5\sigma', 'FontSize', 12, 'HorizontalAlignment', 'left');
    end
    
    % 绘制共享的时间标尺
    time_scale_y = y_bottom_E_NS - vertical_spacing/2;
    time_scale_x_start = time_axis(end) - 60;
    plot([time_scale_x_start, time_axis(end)], [time_scale_y, time_scale_y], 'k', 'LineWidth', 2);
    text(time_scale_x_start + 30, time_scale_y - 2.5, '60s', 'HorizontalAlignment', 'center', 'FontSize', 12);
    
    % 调整X轴范围以容纳新元素
    xlim([time_axis(1), time_axis(end) + 60]);

    disp('  -> 正在保存堆叠图为 .png 和 .fig 格式...');
    saveas(fig, [output_filename_base, '.png']);
    saveas(fig, [output_filename_base, '.fig']);
    close(fig);
end
