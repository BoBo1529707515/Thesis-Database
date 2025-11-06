clear; clc; close all;

% --- 1. 设置文件路径 ---
file_path = "F:\工作文件\RA\数据集\Fig2C\FVB_Panneuronal_Reunion_Isolation_Day3_Mouse#6_(FVB3_2)_preprocessed.mat";

% --- 2. 加载、分箱、Z-score ---
disp("正在加载文件: " + file_path);
try
    load(file_path, 'results_final');
catch ME
    error("文件加载失败！请检查路径是否正确。错误信息: " + ME.message);
end

raw_data = results_final.C_raw_final;
nNeurons = size(raw_data, 1);
num_bins = floor(size(raw_data, 2) / 10);

disp("正在进行1秒分箱...");
bin_data = zeros(nNeurons, num_bins);
for m = 1:nNeurons
    for p = 1:num_bins
        bin_data(m,p) = mean(raw_data(m, ((p*10)-9):(p*10)));
    end
end

disp("正在计算Z-score...");
zscored_data = zscore(bin_data, [], 2);

% --- 3. 计算每个时间点的群体标准差 ---
% 这是核心步骤：沿神经元维度(1)计算标准差，得到每个时间点的离散程度
disp("正在计算群体标准差...");
population_std = std(zscored_data, 0, 1);

% --- 4. 自动寻找事件开始点 ---
% 我们假设前300秒为稳定的基线期
baseline_period = population_std(1:300);
% 定义一个阈值：基线均值 + 3倍基线标准差
threshold = mean(baseline_period) + 3 * std(baseline_period);
% 寻找在基线期之后，信号第一次持续超过阈值的点
% 'find' 返回的是相对于搜索范围的索引，所以要加上基线期长度
potential_start_index = find(population_std(301:end) > threshold, 1, 'first') + 300;

% --- 5. 绘图与可视化 ---
disp("正在绘图...");
figure('Name', 'Population Standard Deviation Analysis', 'Position', [50, 100, 1400, 500]);
plot(1:num_bins, population_std, 'b', 'LineWidth', 1.5);
hold on;

% 绘制基线和阈值线
plot(1:300, baseline_period, 'k', 'LineWidth', 1.5); % 突出显示基线
yline(threshold, 'g--', 'LineWidth', 1, 'Label', '3σ 阈值');

% 标记自动找到的事件开始点
if ~isempty(potential_start_index)
    xline(potential_start_index, 'r--', 'LineWidth', 2, ...
        'Label', ['推测的 Reunion Start ≈ ' num2str(potential_start_index) 's'], ...
        'LabelVerticalAlignment', 'bottom', 'FontSize', 12);
end

% 美化图像
title('群体神经元活动标准差 vs. 时间 (Mouse #2)', 'FontSize', 16);
xlabel('时间 (秒)', 'FontSize', 12);
ylabel('群体Z-score标准差', 'FontSize', 12);
legend({'群体标准差', '基线期'}, 'Location', 'northwest');
grid on;
xlim([0, num_bins]);
ylim([min(population_std)-0.1, max(population_std)+0.1]);

disp('分析完成！请查看生成的图像。');
