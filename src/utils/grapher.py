import numpy as np
import matplotlib.pyplot as plt

def calc_RSRP_2D(eNBs, grid_size_x=100, grid_size_y=50, spacing=10):
    """
    Computes and visualizes RSRP in a 2D area as a heatmap.
    
    :param eNBs: List of base stations, each with (x, y) coordinates and a calc_RSRP() method.
    :param grid_size_x: Number of points along the X-axis.
    :param grid_size_y: Number of points along the Y-axis.
    :param spacing: Defines the spacing (in meters) between points in the grid.
    """

    # Define the 2D grid
    x_range = np.linspace(0, (grid_size_x - 1) * spacing, grid_size_x)  # X-coordinates
    y_range = np.linspace(0, (grid_size_y - 1) * spacing, grid_size_y)  # Y-coordinates
    X, Y = np.meshgrid(x_range, y_range)  # Create 2D grid

    # Initialize RSRP map (set to very low values initially)
    RSRP_map = np.full((grid_size_y, grid_size_x), -150)  # Use (rows, cols) format

    # Compute RSRP for each grid point
    for j in range(grid_size_y):  # Rows (Y-axis)
        for i in range(grid_size_x):  # Columns (X-axis)
            rsrp_values = []  # Store RSRP from all base stations
            for e in eNBs:
                rsrp = e.calculate_received_power((X[j, i], Y[j, i]))  # Fix indexing
                rsrp_values.append(rsrp)

            # Store the strongest RSRP value at this grid point
            RSRP_map[j, i] = max(rsrp_values)

    # Plot the RSRP heatmap
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, RSRP_map, levels=50, cmap="jet")  # Heatmap
    plt.colorbar(label="RSRP (dBm)")  # Color scale
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("2D RSRP Coverage Map")

    # Mark eNB locations
    # for e in eNBs:
    #     plt.scatter(e.location[0], e.location[1], color="blue", marker="^", s=30, label=f"eNB {e.get_id()}")

    # Draw the road as two black horizontal lines at y=240m and y=260m
    plt.axhline(y=240, color='black', linestyle='-', linewidth=2, label="Road Boundary")
    plt.axhline(y=260, color='black', linestyle='-', linewidth=2)

    plt.legend()
    plt.show()
