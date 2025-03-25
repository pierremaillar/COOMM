import numpy as np
import plotly.graph_objects as go
import pandas as pd
from collections import defaultdict
from matplotlib import pyplot as plt
import matplotlib.cm as cm


def plot_3D_ribbons_from_process_solution(solution_df, solution_indices=None, n_points=20, half_width=0.1, n_arrows = 10, save_path="figure/3D_ribbons.png"):
    """
    Plots 3D ribbon structures from a given partitioned solution file using Plotly.

    Parameters:
    -----------
    solution_df : pd.DataFrame
        A DataFrame containing the solution data with columns ['Index_solution', 'X', 'Y', 'Z', 
        'd3x', 'd3y', 'd3z', 'd2x', 'd2y', 'd2z', 's'].
    solution_indices : list of int, optional
        A list of solution indices to be visualized. If None, defaults to [1, 2].
    n_points : int, optional (default=20)
        Number of points across the ribbon width for surface plotting.
    n_arrows : int, optional (default=10)
        Number of arrows representing the cosserat fram allong the centerline to display
    half_width : float, optional (default=0.1)
        Half-width of the ribbon for visualization.
    save_path : str, optional (default="figure/3D_ribbons.png")
        File path to save the output image.

    Returns:
    --------
    fig : plotly.graph_objects.Figure
        A Plotly figure object displaying the 3D ribbon structures.

    Notes:
    ------
    - The function extracts centerline coordinates and constructs a ribbon surface using normal vectors.
    - Direction vectors (d1, d2, d3) are used to define the ribbon width and orientations.
    - Colored ribbons are plotted based on the 's' value in the DataFrame.
    - Arrows representing d1, d2, and d3 are plotted at intervals.
    - The figure is saved as an image and displayed.
    """
    if solution_indices is None:
        print("Don't forget to add the list of index you want to plot.")
        solution_indices = [1,2]

    fig = go.Figure()

    for index in solution_indices:
        one_solution = solution_df[solution_df['Index_solution'] == index]

        X_surf, Y_surf, Z_surf, colors = [], [], [], []

        min_index = min(solution_indices)
        max_index = max(solution_indices)
        opacity = 0.1 + 0.9 * (index - min_index) / (max_index - min_index)

        for i in range(len(one_solution)):
            x, y, z = one_solution.iloc[i][['X', 'Y', 'Z']]
            
            d3 = one_solution.iloc[i][['d3x', 'd3y', 'd3z']].values
            d2 = one_solution.iloc[i][['d2x', 'd2y', 'd2z']].values
            d1 = np.cross(d3, d2)  # d1 is perpendicular to d2 and d3

            # Normalize vectors
            d3 /= np.linalg.norm(d3)
            d2 /= np.linalg.norm(d2)
            d1 /= np.linalg.norm(d1)

            # Create points along the width of the ribbon
            width_points = []
            for j in range(n_points):
                t = (j - n_points / 2) / (n_points / 2)
                width_point = np.array([x, y, z]) + t * half_width * d2
                width_points.append(width_point)

            width_points = np.array(width_points)
            X_surf.append(width_points[:, 0])
            Y_surf.append(width_points[:, 1])
            Z_surf.append(width_points[:, 2])

            color_value = one_solution.iloc[i]['s']
            colors.append([color_value] * n_points)

            if i % int(len(one_solution)/n_arrows) == 0:
                arrow_scale = 0.09
                fig.add_trace(go.Cone(x=[x], y=[y], z=[z], u=[-d1[0]], v=[-d1[1]], w=[-d1[2]], 
                                      colorscale='Blues', anchor='tail', sizemode='absolute',
                                      showscale=False, opacity=opacity, sizeref=arrow_scale, name='d1'))
                fig.add_trace(go.Cone(x=[x], y=[y], z=[z], u=[-d2[0]], v=[-d2[1]], w=[-d2[2]], 
                                      colorscale='Reds', anchor='tail', sizemode='absolute',
                                      showscale=False, opacity=opacity, sizeref=arrow_scale, name='d2'))
                fig.add_trace(go.Cone(x=[x], y=[y], z=[z], u=[d3[0]], v=[d3[1]], w=[d3[2]], 
                                      colorscale='Greens', anchor='tail', sizemode='absolute',
                                      showscale=False, opacity=opacity, sizeref=arrow_scale, name='d3'))

        X_surf, Y_surf, Z_surf, colors = map(np.array, (X_surf, Y_surf, Z_surf, colors))

        print(f"Indice of the solution: {one_solution['Index_solution'].max()}")
        print(f"X_max: {np.max(np.abs(one_solution.X)):.3f}")
        print(f"Y_max: {np.max(np.abs(one_solution.Y)):.3f}")
        print(f"Z_max: {np.max(np.abs(one_solution.Z)):.3f}\n")

        fig.add_trace(go.Surface(x=X_surf, y=Y_surf, z=Z_surf, surfacecolor=colors,
                                 colorscale='Plasma', showscale=False, opacity=opacity, name=f'Ribbon {index}'))

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            xaxis=dict(range=[-0.5, 1]),
            yaxis=dict(range=[-1, 1]),
            zaxis=dict(range=[-1, 1]),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1)
        )
    )
    
    fig.write_image(save_path)
    fig.show()
    return fig


def process_solution_file(path_s):

    # Read the header to extract meta-information
    fl = pd.read_table(path_s, nrows=0, sep='\s+')
    lst = list(fl)
    con = [float(x) for x in lst]
    con = [int(x) for x in con]
    
    ntpl = con[6]  # Points in the time interval [0,1]
    nrowpr = con[8]  # Number of lines printed following the identifying line


    # Read the .s file, skipping bad lines
    sol_raw = pd.read_csv(path_s, sep='\s+', header=None, on_bad_lines='skip', skipinitialspace=True)

    # Search the index of the first line of solutions
    index_no_nan = sol_raw.dropna().index + 1  
    index_no_nan = index_no_nan[sol_raw.iloc[index_no_nan - 1, :].sum(axis=1) > 50]

    # Identify the rows with exactly 1 non-NaN value, likely separator rows
    index_one_non_nan = sol_raw[sol_raw.notna().sum(axis=1) <= 2].index

    # Find sequences of consecutive rows with single non-NaN values
    consecutive_indices = [index_one_non_nan[i] for i in range(len(index_one_non_nan) - 2)
                           if index_one_non_nan[i] == index_one_non_nan[i + 1] - 1
                           and index_one_non_nan[i] == index_one_non_nan[i + 2] - 2]

    consecutive_indices = np.array(consecutive_indices)
    index_no_nan = consecutive_indices - 4*ntpl + 1
    extracted_solutions = []

    # Extract solutions between corresponding pairs of index_no_nan and consecutive_indices
    for i, j in zip(index_no_nan, consecutive_indices):
        extracted_solutions.append(sol_raw.iloc[i:j + 1])

    # Concatenate extracted solutions
    extracted_solutions_df = pd.concat(extracted_solutions)

    # Clean and limit columns in the extracted solutions
    extracted_solutions_cleaned_df = extracted_solutions_df.iloc[:, :7]

    # Flatten and remove NaN values
    flattened = extracted_solutions_cleaned_df.values.flatten()
    non_nan_values = flattened[~np.isnan(flattened)]

    # Define the number of columns (22)
    num_columns = 22

    # Reshape into a DataFrame
    solution_df = pd.DataFrame(non_nan_values.reshape(-1, num_columns))

    # Set column names
    solution_df.columns = ['Cont_par', 'X', 'Y', 'Z', 'd3x', 'd3y', 'd3z', 'd2x', 'd2y', 'd2z', 'd1x', 'd1y', 'd1z',
                           'R1', 'R2', 'R3', 'm1', 'm2', 'm3', 'k2', 'k3', 's']

    # Calculate the number of points per solution
    n = (consecutive_indices[0] - index_no_nan[0] + 1) // 4

    # Determine number of rows
    num_rows = solution_df.shape[0]

    # Create an 'Index_solution' column to label each solution
    index_column = np.repeat(np.arange(1, (num_rows // n) + 2), n)[:num_rows]
    solution_df['Index_solution'] = index_column

    print(f"\nPartition found {solution_df['Index_solution'].max()} solutions with {n} points each")

    return solution_df.astype('float64')


def plot_multiple_solutions(solution_dfs, labels, indices, start_color_idx=0, true_solution=None):
    """
    Plots the solution of various variables as subplots for multiple solution DataFrames, 
    comparing them with the true solution (if provided).

    Parameters:
    -----------
    solution_dfs : list of pandas.DataFrame
        A list of DataFrames containing the solution data. Each DataFrame should have 
        columns: 'Index_solution', 'X', 'Y', 'Z', 'R1', 'R2', 'R3', 'm1', 'm2', 'm3', 'k1', 'k2', 'k3', 's'.
    
    labels : list of str
        Labels corresponding to each solution DataFrame (for legend purposes).
    
    indices : list of int
        A list of indices corresponding to different solutions that need to be plotted.

    start_color_idx : int, optional (default=0)
        An integer defining the starting color index for the colormap.
        Different numbers will shift the color scheme.

    true_solution : pandas.DataFrame, optional
        A DataFrame containing the true solution values for the same variables. 
        If provided, it is plotted for comparison.

    Returns:
    --------
    None
        Displays the plots for the given solutions.
    
    Notes:
    ------
    - Generates subplots for 12 variables: 'X', 'Y', 'Z', 'R1', 'R2', 'R3', 'm1', 'm2', 'm3', 'k1', 'k2', 'k3'.
    - Each subplot compares the values for different solutions.
    - The true solution (if provided) is plotted in red as a dashed line.
    """
    num_solutions = len(solution_dfs)
    
    fig, axs = plt.subplots(4, 3, figsize=(15, 12))
    fig.suptitle('Comparison of Multiple Solutions', fontsize=16)

    variables = [
        ('X', 'X Position'),
        ('Y', 'Y Position'),
        ('Z', 'Z Position'),
        ('R1', 'R1'),
        ('R2', 'R2'),
        ('R3', 'R3'),
        ('m1', 'm1'),
        ('m2', 'm2'),
        ('m3', 'm3'),
        ('k1', 'k1'),
        ('k2', 'k2'),
        ('k3', 'k3')
    ]

    # Define colormaps and choose starting colormap based on start_color_idx
    colormaps = [cm.Blues, cm.Oranges, cm.Greens, cm.Purples, cm.Reds, cm.Greys]
    colormap = colormaps[start_color_idx % len(colormaps)]  # Select colormap based on index
    
    # Generate color gradients for each solution set
    colors = [colormaps[i % len(colormaps)](np.linspace(0.3, 1, len(indices))) for i in range(num_solutions)]

    for i, (var, title) in enumerate(variables):
        row, col = divmod(i, 3)
        axs[row, col].grid(True)

        # Plot each solution DataFrame
        for j, (df, label) in enumerate(zip(solution_dfs, labels)):
            for k, index_solution in enumerate(indices):
                selected_df = df[df['Index_solution'] == index_solution].reset_index()
                color = colors[j][k]  # Assign color based on solution set and index
                
                axs[row, col].plot(
                    selected_df['s'], selected_df[var],
                    color=color, linestyle='-', label=f'{label} (Index {index_solution})'
                )

        # Plot true solution if available
        if true_solution is not None:
            axs[row, col].plot(
                true_solution['s'], true_solution[var],
                color='red', linestyle='--', linewidth=2, label='True Solution'
            )

        axs[row, col].set_ylabel(title, fontsize=12)
        axs[row, col].set_title(title, fontsize=14)
        axs[row, col].tick_params(axis='both', labelsize=10)

    # Add legend to the last subplot
    axs[-1, -1].legend(title='Legend', fontsize=10, loc='upper left')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()


def plot_3D_ribbons_from_process_solutions(solution_df1, solution_indices1, solution_df2, solution_indices2, 
                                           n_points=20, half_width=0.1, n_arrows=10, save_path="figure/3D_ribbons.png"):
    """
    Plots 3D ribbon structures from two different partitioned solution files using Plotly.

    Parameters:
    -----------
    solution_df1 : pd.DataFrame
        First DataFrame containing the solution data with columns ['Index_solution', 'X', 'Y', 'Z', 
        'd3x', 'd3y', 'd3z', 'd2x', 'd2y', 'd2z', 's'].
    
    solution_indices1 : list of int
        List of solution indices to be visualized from solution_df1.

    solution_df2 : pd.DataFrame
        Second DataFrame containing another set of solution data.

    solution_indices2 : list of int
        List of solution indices to be visualized from solution_df2.

    n_points : int, optional (default=20)
        Number of points across the ribbon width for surface plotting.

    n_arrows : int, optional (default=10)
        Number of arrows representing the Cosserat frame along the centerline to display.

    half_width : float, optional (default=0.1)
        Half-width of the ribbon for visualization.

    save_path : str, optional (default="figure/3D_ribbons.png")
        File path to save the output image.

    Returns:
    --------
    fig : plotly.graph_objects.Figure
        A Plotly figure object displaying the 3D ribbon structures.

    Notes:
    ------
    - Two datasets are plotted with separate colors.
    - Opacity varies within each group.
    - Cosserat frame arrows are included.
    """
    fig = go.Figure()

    # Define colormaps and opacity scaling
    datasets = [(solution_df1, solution_indices1, 'Viridis'), (solution_df2, solution_indices2, 'Plasma')]
    
    for solution_df, solution_indices, colormap in datasets:
        min_index = min(solution_indices)
        max_index = max(solution_indices)

        for index in solution_indices:
            one_solution = solution_df[solution_df['Index_solution'] == index]

            X_surf, Y_surf, Z_surf, colors = [], [], [], []
            opacity = 0.2 + 0.8 * (index - min_index) / (max_index - min_index)

            for i in range(len(one_solution)):
                x, y, z = one_solution.iloc[i][['X', 'Y', 'Z']]
                
                d3 = one_solution.iloc[i][['d3x', 'd3y', 'd3z']].values
                d2 = one_solution.iloc[i][['d2x', 'd2y', 'd2z']].values
                d1 = np.cross(d3, d2)  # d1 is perpendicular to d2 and d3

                # Normalize vectors
                d3 /= np.linalg.norm(d3)
                d2 /= np.linalg.norm(d2)
                d1 /= np.linalg.norm(d1)

                # Create points along the width of the ribbon
                width_points = []
                for j in range(n_points):
                    t = (j - n_points / 2) / (n_points / 2)
                    width_point = np.array([x, y, z]) + t * half_width * d2
                    width_points.append(width_point)

                width_points = np.array(width_points)
                X_surf.append(width_points[:, 0])
                Y_surf.append(width_points[:, 1])
                Z_surf.append(width_points[:, 2])

                color_value = one_solution.iloc[i]['s']
                colors.append([color_value] * n_points)

                if i % int(len(one_solution) / n_arrows) == 0:
                    arrow_scale = 0.09
                    fig.add_trace(go.Cone(x=[x], y=[y], z=[z], u=[-d1[0]], v=[-d1[1]], w=[-d1[2]], 
                                          colorscale='Blues', anchor='tail', sizemode='absolute',
                                          showscale=False, opacity=opacity, sizeref=arrow_scale, name=f'd1 ({index})'))
                    fig.add_trace(go.Cone(x=[x], y=[y], z=[z], u=[-d2[0]], v=[-d2[1]], w=[-d2[2]], 
                                          colorscale='Reds', anchor='tail', sizemode='absolute',
                                          showscale=False, opacity=opacity, sizeref=arrow_scale, name=f'd2 ({index})'))
                    fig.add_trace(go.Cone(x=[x], y=[y], z=[z], u=[d3[0]], v=[d3[1]], w=[d3[2]], 
                                          colorscale='Greens', anchor='tail', sizemode='absolute',
                                          showscale=False, opacity=opacity, sizeref=arrow_scale, name=f'd3 ({index})'))

            X_surf, Y_surf, Z_surf, colors = map(np.array, (X_surf, Y_surf, Z_surf, colors))

            print(f"Solution {index} from dataset ({colormap}):")
            print(f"  X_max: {np.max(np.abs(one_solution.X)):.3f}")
            print(f"  Y_max: {np.max(np.abs(one_solution.Y)):.3f}")
            print(f"  Z_max: {np.max(np.abs(one_solution.Z)):.3f}\n")

            fig.add_trace(go.Surface(
                x=X_surf, y=Y_surf, z=Z_surf, surfacecolor=colors,
                colorscale=colormap, showscale=False,
                opacity=opacity, name=f'Ribbon {index}'
            ))

    fig.update_layout(
        title="3D Ribbon Comparison from Two Datasets",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            xaxis=dict(range=[-0.5, 1]),
            yaxis=dict(range=[-1, 1]),
            zaxis=dict(range=[-1, 1]),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1)
        )
    )
    
    fig.write_image(save_path)
    fig.show()
    return fig

def process_solution_data(pp_list_read, step_skip, base_length):
    """
    Processes solution data and converts it into a structured DataFrame.

    Parameters:
    -----------
    pp_list_read : dict
        Dictionary containing solution data with keys: 'time', 'step', 'position', 
        'directors', 'internal_stress', 'internal_couple', and 'curvature'.
    
    step_skip : int
        Step increment to normalize the solution index.

    base_length : float
        Reference length to normalize positional values.

    Returns:
    --------
    pandas.DataFrame
        Processed DataFrame containing the solution data.
    """
    rows = []
    j = 0

    for t, step, pos, director, stress, couple, curvature in zip(
        pp_list_read["time"], pp_list_read["step"],
        pp_list_read["position"], pp_list_read["directors"],
        pp_list_read["internal_stress"], pp_list_read["internal_couple"],
        pp_list_read["curvature"]
    ):
        num_elements = pos.shape[1]  # Number of elements

        # Extend director matrix
        last_element = director[:, :, -1][:, :, np.newaxis] 
        director_extended = np.concatenate((director, last_element), axis=2)

        # Extend couple
        last_element = couple[:, -1][:, np.newaxis] 
        couple_extended = np.concatenate((couple, last_element), axis=1)
        couple_extended = np.concatenate((couple_extended, last_element), axis=1)

        # Extend stress
        last_element = stress[:, -1][:, np.newaxis] 
        stress_extended = np.concatenate((stress, last_element), axis=1)

        # Extend curvature
        last_element = curvature[:, -1][:, np.newaxis] 
        curvature_extended = np.concatenate((curvature, last_element), axis=1)
        curvature_extended = np.concatenate((curvature_extended, last_element), axis=1)

        # Populate rows
        for i in range(num_elements):
            rows.append({
                "Index_solution": j,
                "time": t,
                "step": step,
                "s": i / (num_elements - 1),
                "X": pos[0, i] / base_length,
                "Y": pos[1, i] / base_length,
                "Z": pos[2, i] / base_length,
                'd3x': director_extended[2, 0, i], 
                'd3y': director_extended[2, 1, i], 
                'd3z': director_extended[2, 2, i],
                'd2x': director_extended[1, 0, i], 
                'd2y': director_extended[1, 1, i], 
                'd2z': director_extended[1, 2, i],
                'R1': stress_extended[0, i],
                'R2': stress_extended[1, i],
                'R3': stress_extended[2, i],
                'm1': couple_extended[2, i],
                'm2': couple_extended[2, i],
                'm3': couple_extended[2, i],
                'k1': curvature_extended[2, i],
                'k2': curvature_extended[2, i],
                'k3': curvature_extended[2, i],            
            })
        j+=1

    return pd.DataFrame(rows)