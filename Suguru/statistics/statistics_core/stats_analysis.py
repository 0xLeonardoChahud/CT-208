import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr, pearsonr
from typing import Dict, List, Any

def stats(solver_results: Dict[str, List[Any]], timeout_value: float = 60.0) -> pd.DataFrame:
    """
    Performs robust analysis on Suguru Simulated Annealing results, including
    feature engineering and comprehensive correlation visualization.

    Args:
        solver_results (Dict[str, List[Any]]): A dictionary containing 
            the raw experiment data (e.g., 'solved', 'elapsed', 'size').
        timeout_value (float): The maximum allowed time (in seconds) for a run.
            This value will replace np.inf for calculation purposes. Defaults to 60.0.

    Returns:
        pd.DataFrame: The processed DataFrame containing the results and new features.
    """
    
    # --- 1. DataFrame Creation and Feature Engineering ---
    print("--- Starting Data Processing and Feature Engineering ---")
    df = pd.DataFrame(solver_results)
    
    # 1.1 Handle np.inf Timeouts and convert 'solved'
    # Replaces np.inf with the defined timeout_value
    df.loc[df['elapsed'] == np.inf, 'elapsed'] = timeout_value
    df['solved'] = df['solved'].astype(bool)
    
    # 1.2 Feature Engineering (Performed on the MAIN DataFrame: df)
    # A. Density of Tips: Tips relative to the total number of cells
    df['tip_density'] = df['tips'] / df['size']
    
    # B. Region Complexity: Size / Region Count (measures average cell size per region)
    df['region_complexity'] = df['size'] / df['region_count']
    
    # C. Search Space Proxy: A feature combining size and regions (log-scaled)
    df['log_difficulty_proxy'] = np.log(df['size'] * df['region_count'])
    
    # 1.3 Create the FILTERED DataFrame (df_solved) AFTER Feature Engineering
    # This is the crucial step to ensure df_solved has all new columns
    df_solved = df[df['solved'] == True].copy()
    
    total_runs = len(df)
    solved_runs = len(df_solved)
    
    print(f"Total Runs: {total_runs}, Solved Runs: {solved_runs}")
    print("-" * 30)
    
    # --- 2. Key Metrics Calculation ---
    print("--- Key Performance Metrics ---")
    success_rate = solved_runs / total_runs
    print(f"1. Overall Success Rate: {success_rate:.2f} ({solved_runs}/{total_runs})")
    
    if solved_runs > 0:
        avg_time_solved = df_solved['elapsed'].mean()
        print(f"2. Avg Solving Time (Solved Only): {avg_time_solved:.3f} s")
    
    print("-" * 30)

    # --- 3. Robust Correlation Analysis ---
    
    print("--- Robust Correlation Analysis ---")
    
    features_to_correlate = ['size', 'tips', 'region_count', 
                             'tip_density', 'region_complexity', 'log_difficulty_proxy']
    
    # A. Correlation with SUCCESS RATE (Point-Biserial)
    correlation_results_success = {}
    solved_numeric = df['solved'].astype(int) 
    
    for feature in features_to_correlate:
        r, p = pointbiserialr(df[feature], solved_numeric)
        correlation_results_success[feature] = {'r_success': r, 'p_value': p}
        
    corr_success_df = pd.DataFrame.from_dict(correlation_results_success, orient='index')
    corr_success_df = corr_success_df.sort_values(by='r_success', ascending=False)
    
    print("Point-Biserial Correlation (r) with SUCCESS RATE:")
    print("(* Positive 'r' means higher feature value is correlated with success.)")
    print(corr_success_df.to_string())
    
    print("\n" + "-" * 30)
    
    # B. Correlation with ELAPSED TIME (Pearson, Solved Puzzles Only)
    if solved_runs > 0:
        correlation_results_time = {}
        for feature in features_to_correlate:
            # This is the line that required df_solved to have the new features
            r, p = pearsonr(df_solved[feature], df_solved['elapsed']) 
            correlation_results_time[feature] = {'r_time': r, 'p_value': p}
            
        corr_time_df = pd.DataFrame.from_dict(correlation_results_time, orient='index')
        corr_time_df = corr_time_df.sort_values(by='r_time', ascending=False)
        
        print("Pearson Correlation (r) with ELAPSED TIME (Solved Only):")
        print("(* Positive 'r' means higher feature value is correlated with longer time.)")
        print(corr_time_df.to_string())
    
    print("-" * 30)
    
    # --- 4. Correlation Visualizations ---
    
    sns.set_style("whitegrid")
    
    # A. Heatmap of All Features and Outcomes
    heatmap_cols = ['elapsed', 'solved'] + features_to_correlate
    corr_matrix_full = df[heatmap_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix_full, 
                annot=True, 
                cmap='viridis', 
                fmt=".2f", 
                linewidths=.5,
                cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Correlation Heatmap of All Features and Outcomes')
    plt.show()

    # B. Pair Plot
    pair_plot_features = ['elapsed', 'solved', 'tip_density', 'region_complexity']
    
    print("\nGenerating Pair Plot...")
    
    pair_plot = sns.pairplot(df, 
                             vars=[f for f in pair_plot_features if f != 'solved'], 
                             hue='solved', 
                             palette={True: 'green', False: 'red'}, 
                             diag_kind='kde')
    
    pair_plot.fig.suptitle('Pair Plot of Key Features (Grouped by Success/Failure)', y=1.02)
    plt.show()
    
    # --- 5. Other Visualizations (Originals) ---
    
    # A. Performance Scatter Plots (Timeouts Capped)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    plt.suptitle('Performance Metrics of SA Solver by Puzzle Features (Timeouts Capped)', fontsize=16, y=1.02)

    sns.scatterplot(x='size', y='elapsed', hue='solved', data=df, ax=axes[0], palette={True: 'green', False: 'red'}, alpha=0.7)
    axes[0].set_title('Elapsed Time vs. Grid Area (Size)')

    sns.scatterplot(x='region_count', y='elapsed', hue='solved', data=df, ax=axes[1], palette={True: 'green', False: 'red'}, alpha=0.7)
    axes[1].set_title('Elapsed Time vs. Region Count')

    sns.scatterplot(x='tips', y='elapsed', hue='solved', data=df, ax=axes[2], palette={True: 'green', False: 'red'}, alpha=0.7)
    axes[2].set_title('Elapsed Time vs. Initial Tips')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # B. Time Distribution and Group Comparison (Solved Only)
    if solved_runs > 0:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        sns.histplot(df_solved['elapsed'], bins=15, kde=True, color='skyblue', ax=axes[0])
        axes[0].set_title('Distribution of Elapsed Time for Solved Puzzles')
        axes[0].set_xlabel('Elapsed Time (s)')
        
        # Box Plot of Time by Puzzle Size 
        sns.boxplot(x='size', y='elapsed', data=df_solved, color='lightcoral', ax=axes[1])
        axes[1].set_title('Elapsed Time Distribution by Puzzle Grid Area')
        axes[1].set_xlabel('Grid Area (Size)')
        
        plt.tight_layout()
        plt.show()
        
    print("-" * 30)
    print("--- Analysis Complete ---")
    return df