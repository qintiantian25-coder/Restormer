%% ================================================================
%  ceshi.m — 无盲元处理的全图生成
%
%  管线: NUC → 多项式条纹抑制 → 对比度增强 (无任何盲元检测/修复)
%  输出: ceshi_full.png (6000×6000)
% ================================================================

clc; clear; close all; tic;

base_path = 'E:\DD数据';
mat_path  = fullfile(base_path, '非均匀校正系数', 'YD_SW_A_D_72000_4hz_xs.mat');
raw_path  = fullfile(base_path, 'YD背景100帧', '引导短波_5_2_12_20260116170613488.raw');

%% [1/3] NUC
fprintf('=== [1/3] NUC ===\n');
calib = load(mat_path);
K = double(calib.kk); B = double(calib.bb); K(K < 0.5) = 0.1;
fid = fopen(raw_path, 'rb'); frame = fread(fid, [6000, 6000], 'uint16'); fclose(fid);
data_nuc = (double(frame) - B) ./ K;
[rows, cols] = size(data_nuc);

%% [2/3] 条纹抑制 + 对比度增强
fprintf('=== [2/3] 条纹抑制 + 对比度 ===\n');

% 多项式条纹抑制 (与 mangyuan.m 一致)
col_means = mean(data_nuc, 1);
x_axis = 1:length(col_means);
p = polyfit(x_axis, col_means, 3);
data_nuc = data_nuc - repmat(col_means - polyval(p, x_axis), rows, 1);

row_means = mean(data_nuc, 2);
y_axis = (1:length(row_means)).';
p_row = polyfit(y_axis, row_means, 3);
data_nuc = data_nuc - repmat(row_means - polyval(p_row, y_axis), 1, cols);

% 对比度增强
low_val  = quantile(data_nuc(:), 0.001);
high_val = quantile(data_nuc(:), 0.999);
data_nuc = (data_nuc - low_val) / (high_val - low_val);
data_nuc(data_nuc < 0) = 0; data_nuc(data_nuc > 1) = 1;
data_nuc = data_nuc .^ 0.6;

%% [3/3] 输出
fprintf('=== [3/3] 输出 ===\n');
imwrite(uint8(data_nuc * 255), fullfile(base_path, 'ceshi_full.png'));
fprintf('  → ceshi_full.png (6000×6000)\n');
fprintf('\n=== 完成 (%.0fs) ===\n', toc);
