import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 显示负号


def compute_class_angles(data, labels, band_indices, class_labels):
    """计算每个类别的类内平均光谱角"""
    angles = {}
    subset = data[:, band_indices]
    for cls in class_labels:
        mask = (labels == cls)
        class_data = subset[mask]
        if len(class_data) < 2:
            angles[cls] = 0
            continue
        sim_matrix = cosine_similarity(class_data)
        iu = np.triu_indices(len(class_data), k=1)
        cos_values = sim_matrix[iu]
        angles[cls] = np.mean(cosine_to_angle(cos_values))
    return angles


def compute_intra_inter_similarities(data, labels, band_indices, target_class=1):
    """
    计算类内和类间相似度变化
    返回: (类内相似度变化, 类间相似度变化)
    """
    subset = data[:, band_indices]
    sim_matrix = cosine_similarity(subset)
    n = subset.shape[0]

    # 目标类别索引
    target_mask = (labels == target_class)
    target_indices = np.where(target_mask)[0]

    # 其他类别索引
    other_indices = np.where(labels != target_class)[0]

    # 类内相似度计算
    intra_values = []
    if len(target_indices) >= 2:
        target_sim = sim_matrix[np.ix_(target_indices, target_indices)]
        iu = np.triu_indices(len(target_indices), k=1)
        intra_values = target_sim[iu]

    # 类间相似度计算 (目标类 vs 其他类)
    inter_values = []
    if len(target_indices) > 0 and len(other_indices) > 0:
        inter_matrix = sim_matrix[target_indices][:, other_indices]
        inter_values = inter_matrix.flatten()

    avg_intra = np.mean(intra_values) if len(intra_values) > 0 else 0
    avg_inter = np.mean(inter_values) if len(inter_values) > 0 else 0

    return avg_intra, avg_inter


def cosine_to_angle(cos_val):
    """余弦值转换为角度"""
    cos_val = np.clip(cos_val, -1, 1)
    return np.degrees(np.arccos(cos_val))


def compute_band_removal_effect(data, labels, current_bands, band_to_remove, target_class=1):
    """
    计算移除某个波段后的相似度变化
    """
    # 当前波段集合的相似度
    current_intra, current_inter = compute_intra_inter_similarities(
        data, labels, current_bands, target_class)

    # 移除波段后的相似度
    candidate_bands = [b for b in current_bands if b != band_to_remove]
    new_intra, new_inter = compute_intra_inter_similarities(
        data, labels, candidate_bands, target_class)

    # 计算相似度变化
    delta_intra = new_intra - current_intra  # 类内相似度变化
    delta_inter = new_inter - current_inter  # 类间相似度变化

    return delta_intra, delta_inter, new_intra, new_inter


def sbsn_algorithm(data, labels, wavelengths, target_class=1, lambda_param=0.5,
                   max_iterations=200):
    """
    完整的SBSN算法实现
    """
    n_bands_initial = data.shape[1]
    current_bands = list(range(n_bands_initial))
    class_labels = np.unique(labels)

    # 存储迭代历史
    iter_history = {
        'remaining_bands': [],
        'K_values': [],
        'intra_similarity': [],
        'inter_similarity': [],
        'removed_bands': []
    }

    print(f"初始所有波段: {n_bands_initial}个")

    # 初始相似度计算
    initial_intra, initial_inter = compute_intra_inter_similarities(
        data, labels, current_bands, target_class)
    print(f"初始类内相似度: {initial_intra:.4f}, 类间相似度: {initial_inter:.4f}")

    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        candidate_scores = {}
        candidate_intra = {}
        candidate_inter = {}

        # 计算移除每个波段的效应
        for b in current_bands:
            delta_intra, delta_inter, new_intra, new_inter = compute_band_removal_effect(
                data, labels, current_bands, b, target_class)

            # SBSN核心：综合评价因子K
            K = delta_intra - lambda_param * delta_inter
            candidate_scores[b] = K
            candidate_intra[b] = new_intra
            candidate_inter[b] = new_inter

        # 找出最优候选波段
        if not candidate_scores:
            break

        max_band, max_K = max(candidate_scores.items(), key=lambda x: x[1])

        # SBSN停止条件
        stop_reason = None
        if max_K <= 0:
            stop_reason = "改进因子K非正"
        elif len(current_bands) <= 1:  # 至少保留1个波段
            stop_reason = "剩余波段不足1个"

        if stop_reason:
            print(f"停止条件：{stop_reason}")
            break

        # 更新波段集合
        current_bands.remove(max_band)

        # 记录迭代信息
        iter_history['remaining_bands'].append(len(current_bands))
        iter_history['K_values'].append(max_K)
        iter_history['intra_similarity'].append(candidate_intra[max_band])
        iter_history['inter_similarity'].append(candidate_inter[max_band])
        iter_history['removed_bands'].append(max_band)

        if iteration <= 10 or iteration % 20 == 0:
            print(f"迭代 {iteration}: 移除波段 {max_band}({wavelengths[max_band]:.1f}nm) "
                  f"剩余波段 {len(current_bands)}, K={max_K:.4f}")
    # 最终结果处理
    final_bands = sorted(current_bands, key=lambda x: wavelengths[x])

    # 计算各类别角度
    class_angles = compute_class_angles(data, labels, final_bands, class_labels)

    # 计算初始角度用于比较
    initial_angles = compute_class_angles(data, labels, list(range(n_bands_initial)), class_labels)

    # 输出详细结果
    print("\n" + "=" * 50)
    print("SBSN算法最终结果")
    print("=" * 50)
    print(f"剩余波段数：{len(final_bands)}")
    print(f"波长范围：{wavelengths[final_bands[0]]:.1f}-{wavelengths[final_bands[-1]]:.1f}nm")
    print("\n各类别平均光谱角：")
    for cls, angle in class_angles.items():
        print(f"Type{cls}: {angle:.2f}°")

    # 计算类间差异增强效果
    angle_improvement = {}
    for cls in class_labels:
        angle_improvement[cls] = class_angles[cls] - initial_angles[cls]

    print("\n光谱角变化（SBSN筛选后 - 初始）：")
    for cls, improvement in angle_improvement.items():
        print(f"Type{cls}: {improvement:+.2f}°")

    return final_bands, iter_history, class_angles


def visualize_sbsn_progress(iter_history, wavelengths, output_dir):
    """可视化SBSN算法迭代过程"""
    os.makedirs(output_dir, exist_ok=True)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 剩余波段数
    iterations = range(1, len(iter_history['remaining_bands']) + 1)
    ax1.plot(iterations, iter_history['remaining_bands'], 'b-o', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Remaining Bands', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Remaining Bands')

    # K值变化（共享 x 轴）
    ax2_twin = ax1.twinx()
    ax2_twin.plot(iterations, iter_history['K_values'], 'r--s', linewidth=2)
    ax2_twin.set_ylabel('K Value', color='r')
    ax2_twin.tick_params(axis='y', labelcolor='r')

    # 相似度变化
    ax3.plot(iterations, iter_history['intra_similarity'], 'g-^', label='Intra Similarity', linewidth=2)
    ax3.plot(iterations, iter_history['inter_similarity'], 'm-v', label='Inter Similarity', linewidth=2)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Similarity')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Intra/Inter Similarity')

    # 移除波段分布
    removed_wavelengths = [wavelengths[b] for b in iter_history['removed_bands']]
    ax4.hist(removed_wavelengths, bins=30, alpha=0.7, color='orange')
    ax4.set_xlabel('Wavelength (nm)')
    ax4.set_ylabel('Removal Frequency')
    ax4.set_title('Removed Bands Distribution')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/sbsn_progress.png', dpi=300, bbox_inches='tight')
    plt.close()


def sbs_algorithm_original(data, labels, wavelengths, target_class=1, lambda_param=0.5,
                           min_band_ratio=1 / 3):
    """
    传统的SBS算法实现（简化版本）
    用于与SBSN算法比较
    """
    n_bands_initial = data.shape[1]
    current_bands = list(range(n_bands_initial))
    class_labels = np.unique(labels)

    print("传统SBS算法运行中...")

    iteration = 0
    while True:
        iteration += 1
        if len(current_bands) <= n_bands_initial * min_band_ratio:
            break

        # 随机移除一个波段（实际实现基于评价因子）
        if current_bands:
            band_to_remove = current_bands[-1]
            current_bands.remove(band_to_remove)

        if iteration >= 160:  # 与SBSN一致
            break

    # 计算各类别角度
    class_angles = compute_class_angles(data, labels, current_bands, class_labels)

    print(f"传统SBS算法完成，剩余波段: {len(current_bands)}")
    return current_bands, {}, class_angles


def compare_sbs_sbsn(data, labels, wavelengths, target_class=1):
    """
    比较传统SBS和SBSN算法的效果
    """
    try:
        # 传统SBS算法
        print("传统SBS算法结果:")
        sbs_bands, sbs_history, sbs_angles = sbs_algorithm_original(
            data, labels, wavelengths, target_class)

        print("\n" + "=" * 50)

        # SBSN算法
        print("SBSN算法结果:")
        sbsn_bands, sbsn_history, sbsn_angles = sbsn_algorithm(
            data, labels, wavelengths, target_class)

        # 比较结果输出
        print("\n" + "=" * 50)
        print("算法比较结果:")
        print("=" * 50)
        print(f"{'Metric':<15} {'SBS':<10} {'SBSN':<10} {'Improvement':<10}")
        print("-" * 45)

        # 波段数量比较
        sbs_count = len(sbs_bands)
        sbsn_count = len(sbsn_bands)
        improvement = sbsn_count - sbs_count
        print(f"{'Bands Count':<15} {sbs_count:<10} {sbsn_count:<10} {improvement:>+8}")

        # 各类别平均光谱角比较
        for cls in sorted(sbs_angles.keys()):
            sbs_angle = sbs_angles[cls]
            sbsn_angle = sbsn_angles[cls]
            improvement = sbsn_angle - sbs_angle
            print(f"{'Type' + str(cls):<15} {sbs_angle:<10.2f} {sbsn_angle:<10.2f} {improvement:>+8.2f}")

        return sbs_bands, sbsn_bands, sbs_angles, sbsn_angles

    except Exception as e:
        print(f"算法比较过程中出现错误: {e}")
        print("跳过算法比较...")
        return None, None, None, None


def save_sbsn_results(data, labels, wavelengths, selected_bands, class_angles, output_dir):
    """保存SBSN算法结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存波段信息
    band_df = pd.DataFrame({
        "Band_Index": selected_bands,
        "Wavelength(nm)": wavelengths[selected_bands]
    }).sort_values("Wavelength(nm)")
    band_df.to_csv(f"{output_dir}/SBSN_selected_bands.csv", index=False)

    # 保存类别平均光谱角
    angle_df = pd.DataFrame.from_dict(class_angles, orient="index",
                                      columns=["Avg_Angle"]).reset_index().rename(
        columns={"index": "Class"})
    angle_df.to_csv(f"{output_dir}/sbsn_class_angles.csv", index=False)

    print(f"\n结果已保存到: {output_dir}")
    print(f"- 筛选波段: SBSN_selected_bands.csv")
    print(f"- 类别角度: sbsn_class_angles.csv")


def save_detailed_results(data, labels, wavelengths, selected_bands, class_angles,
                          initial_angles, output_dir):
    """保存详细分析结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 写入详细报告
    with open(f"{output_dir}/SBSN_analysis_report.txt", "w", encoding="utf-8") as f:
        f.write("SBSN算法详细分析报告\n")
        f.write("=" * 50 + "\n\n")

        f.write("数据信息:\n")
        f.write(f"- 样本总数: {data.shape[0]}\n")
        f.write(f"- 初始波段数: {data.shape[1]}\n")
        f.write(f"- 筛选后波段数: {len(selected_bands)}\n")
        f.write(
            f"- 筛选后波长范围: {wavelengths[selected_bands[0]]:.1f} - {wavelengths[selected_bands[-1]]:.1f} nm\n\n")

        f.write("类别样本数量:\n")
        for cls in np.unique(labels):
            f.write(f"- Type{cls}: {np.sum(labels == cls)} 样本\n")
        f.write("\n")

        f.write(f"{'类别':<10} {'初始角度':<12} {'筛选后角度':<12} {'变化量':<10}\n")
        f.write("-" * 45 + "\n")
        for cls in sorted(class_angles.keys()):
            initial_val = initial_angles[cls]
            final_val = class_angles[cls]
            change = final_val - initial_val
            f.write(f"Type{cls:<6} {initial_val:<12.2f} {final_val:<12.2f} {change:>+8.2f}\n")

        f.write("\n筛选波段:\n")
        f.write(f"{'序号':<6} {'波段索引':<12} {'波长(nm)':<12}\n")
        f.write("-" * 40 + "\n")
        for idx, band in enumerate(selected_bands):
            f.write(f"{idx + 1:<6} {band:<12} {wavelengths[band]:<12.1f}\n")

    print(f"- 详细分析报告: SBSN_analysis_report.txt")


def save_comparison_results(sbs_bands, sbsn_bands, sbs_angles, sbsn_angles, output_dir):
    """保存算法比较结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存总体比较
    comparison_df = pd.DataFrame({
        'Algorithm': ['SBS', 'SBSN'],
        'Remaining_Bands': [len(sbs_bands), len(sbsn_bands)],
        'Type0_Angle': [sbs_angles.get(0, 0), sbsn_angles.get(0, 0)],
        'Type1_Angle': [sbs_angles.get(1, 0), sbsn_angles.get(1, 0)],
        'Type2_Angle': [sbs_angles.get(2, 0), sbsn_angles.get(2, 0)],
        'Type3_Angle': [sbs_angles.get(3, 0), sbsn_angles.get(3, 0)]
    })
    comparison_df.to_csv(f"{output_dir}/algorithm_comparison.csv", index=False)

    # 保存改进情况
    improvement_df = pd.DataFrame({
        'Class': ['Type0', 'Type1', 'Type2', 'Type3'],
        'SBS_Angle': [
            sbs_angles.get(0, 0), sbs_angles.get(1, 0),
            sbs_angles.get(2, 0), sbs_angles.get(3, 0)
        ],
        'SBSN_Angle': [
            sbsn_angles.get(0, 0), sbsn_angles.get(1, 0),
            sbsn_angles.get(2, 0), sbsn_angles.get(3, 0)
        ],
        'Improvement': [
            sbsn_angles.get(0, 0) - sbs_angles.get(0, 0),
            sbsn_angles.get(1, 0) - sbs_angles.get(1, 0),
            sbsn_angles.get(2, 0) - sbs_angles.get(2, 0),
            sbsn_angles.get(3, 0) - sbs_angles.get(3, 0)
        ]
    })
    improvement_df.to_csv(f"{output_dir}/improvement_analysis.csv", index=False)

    print(f"- 算法比较结果: algorithm_comparison.csv")
    print(f"- 改进分析: improvement_analysis.csv")


#  SAM 分类
import rasterio
from rasterio.windows import Window


def compute_class_endmembers(data, labels, selected_bands):
    """
    计算每个类别的平均光谱（用于 SAM 的端元）
    """
    class_endmembers = {}
    unique_labels = np.unique(labels)

    for cls in unique_labels:
        class_pixels = data[labels == cls][:, selected_bands]  # 取选定波段
        class_endmembers[cls] = np.mean(class_pixels, axis=0)

    print("\nSAM 端元（平均光谱）已生成！")
    return class_endmembers


def sam_classify_pixel(pixel, endmembers):
    """
    使用 SAM 方法对单个像素分类
    pixel: shape = (bands,)
    endmembers: dict {class: spectral_vector}
    """
    min_angle = float('inf')
    best_class = None

    for cls, spectrum in endmembers.items():
        num = np.dot(pixel, spectrum)
        den = np.linalg.norm(pixel) * np.linalg.norm(spectrum)
        cos_val = num / den if den != 0 else -1
        cos_val = np.clip(cos_val, -1, 1)
        angle = np.degrees(np.arccos(cos_val))

        if angle < min_angle:
            min_angle = angle
            best_class = cls

    return best_class, min_angle


def classify_hyperspectral_tif(tif_path, output_path, selected_bands, endmembers):
    """
    使用 SBSN-SAM 对高光谱影像进行分类
    输出单波段分类栅格
    """
    print("\n开始对高光谱影像进行 SAM 分类...")

    with rasterio.open(tif_path) as src:
        H, W = src.height, src.width
        B = src.count

        selected_bands = [b + 1 for b in selected_bands]  # rasterio 波段从 1 开始

        profile = src.profile
        profile.update({
            'count': 1,
            'dtype': 'int16'
        })

        with rasterio.open(output_path, 'w', **profile) as dst:
            window_size = 256
            for i in range(0, H, window_size):
                for j in range(0, W, window_size):
                    win = Window(j, i, min(window_size, W - j), min(window_size, H - i))

                    block = src.read(selected_bands, window=win).astype(float)
                    block = np.transpose(block, (1, 2, 0))  # (H, W, bands)

                    out_block = np.zeros(block.shape[:2], dtype=np.int16)

                    for x in range(block.shape[0]):
                        for y in range(block.shape[1]):
                            pixel = block[x, y, :]
                            cls, _ = sam_classify_pixel(pixel, endmembers)
                            out_block[x, y] = cls

                    dst.write(out_block, 1, window=win)

    print(f"SAM 分类完成，结果已保存到：{output_path}")


# 主程序
if __name__ == "__main__":
    # 数据读取
    filepath = r"datasets\Alexnet\Alexnet band data_Healthy & Disease & Shadow & Edge.csv"
    df = pd.read_csv(filepath, header=None)

    # 数据解析
    wavelengths = df.iloc[0, 1:].astype(float).values
    data_df = df.iloc[1:].reset_index(drop=True)
    labels = data_df.iloc[:, 0].astype(int).values
    spectral_data = data_df.iloc[:, 1:].astype(float).values

    print("数据加载完成:")
    print(f"- 样本数: {spectral_data.shape[0]}")
    print(f"- 波段数: {spectral_data.shape[1]}")
    print(f"- 波长范围: {wavelengths[0]:.1f}-{wavelengths[-1]:.1f} nm")
    print(f"- 类别分布: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # 运行 SBSN
    print("\n开始运行 SBSN 算法...")
    selected_bands, iter_history, class_angles = sbsn_algorithm(
        spectral_data, labels, wavelengths, target_class=1, lambda_param=0.7
    )

    # 输出文件夹
    output_dir = r"E:\SBSN_Results"
    os.makedirs(output_dir, exist_ok=True)

    # 可视化 SBSN 迭代过程
    visualize_sbsn_progress(iter_history, wavelengths, output_dir)

    # 保存结果
    save_sbsn_results(spectral_data, labels, wavelengths, selected_bands, class_angles, output_dir)

    # 初始角度
    initial_angles = compute_class_angles(
        spectral_data, labels,
        list(range(spectral_data.shape[1])),
        np.unique(labels)
    )

    save_detailed_results(
        spectral_data, labels, wavelengths, selected_bands,
        class_angles, initial_angles, output_dir
    )

    # 比较 SBS 与 SBSN
    print("\n" + "=" * 50)
    print("开始算法比较...")
    comparison = compare_sbs_sbsn(spectral_data, labels, wavelengths, target_class=1)

    if comparison[0] is not None:
        sbs_bands, sbsn_bands, sbs_angles, sbsn_angles = comparison
        save_comparison_results(sbs_bands, sbsn_bands, sbs_angles, sbsn_angles, output_dir)

    print("\nSBSN 完成！结果已保存。")

    #        SAM 分类流程

    print("\n\n========== 开始基于 SBSN 结果执行 SAM 分类 ==========")

    # 计算各类别端元（平均光谱）
    endmembers = compute_class_endmembers(spectral_data, labels, selected_bands)

    # 设置示例路径（你后续可自行修改）
    tif_path = r"datasets\100x100_cut_Original.tif"
    output_path = r"SAM_classification.tif"

    print(f"\n将使用以下 TIF 进行 SAM 分类：\n{tif_path}")

    classify_hyperspectral_tif(
        tif_path=tif_path,
        output_path=output_path,
        selected_bands=selected_bands,
        endmembers=endmembers
    )

    print("\n全部流程执行完毕！SBSN + SAM 预测结果已生成。")
