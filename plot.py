import mplsoccer
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

def dark_theme():
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['font.weight'] = 'bold'
    mpl.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
    mpl.rcParams['text.color'] = 'white'
    return 'white', 'black'

def light_theme():
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['font.weight'] = 'bold'
    mpl.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
    mpl.rcParams['text.color'] = 'black'
    return 'black', 'white'

def plot_pitch(df, theme='light', player_col='red', alpha=0.5):
    """Original plot_pitch function - unchanged for backward compatibility"""
    if theme == 'light':
        point_color, bg_color = light_theme()
    elif theme == 'dark':
        point_color, bg_color = dark_theme()
    else:
        point_color, bg_color = light_theme()
        
    gold_col = '#ad993c'
    in_col = '#868686'

    fig, ax = plt.subplots()
    fig.set_facecolor(bg_color)
    pitch = mplsoccer.VerticalPitch(pitch_type='opta', pitch_color=bg_color,
                        linewidth=1.5, line_alpha=0.5, goal_alpha=0.5, half=True)
    pitch.draw(ax=ax)

    if len(df) == 0:
        return fig, ax

    # SCATTER POINTS
    goal_cond = df['Shot Result']=='Goal'
    pitch.scatter(df[~goal_cond]['x']*100, df[~goal_cond]['y']*100, c='none', ec=point_color, alpha=0.15, ax=ax)
    pitch.scatter(df[goal_cond]['x']*100, df[goal_cond]['y']*100, c='none', ec=player_col, hatch='////////', alpha=alpha, ax=ax)

    # INTERNAL DATA
    player = df['Player'].iloc[0] if 'Player' in df.columns else "Unknown"
    start_year = df['Season'].min() if 'Season' in df.columns else 2024
    end_year = df['Season'].max() if 'Season' in df.columns else 2024
    avg_x, avg_y = df['x'].mean()*100, df['y'].mean()*100
    total_goals = sum(goal_cond)
    total_shots = len(df)
    total_xg = df['Shot xG'].sum()
    avg_distance_to_goal = np.sqrt(np.sum(np.square(np.array([100, 50])-
                                                    np.array([avg_x, avg_y]))))
    
    # DISTANCE
    # circle = patches.Circle((50, 100), avg_distance_to_goal, ec=point_color, fc='None', alpha=0.5, ls=':', lw=1.5)
    # ax.add_patch(circle)
    # ax.annotate("", xy=(50-avg_distance_to_goal, 101.5), xycoords='data', 
    #             xytext=(50, 101.5), textcoords='data',
    #             size=10, arrowprops=dict(arrowstyle="-|>",fc='none', ls='--', ec=player_col))
    # ax.text(49-avg_distance_to_goal, 101.5, "{:.2f}m".format(avg_distance_to_goal), size=10, 
    #         ha='left', va='center',c=in_col)
    # ax.text(51, 101.5, "Avg distance", size=8, ha='right', va='center')
    # DISTANCE
    circle = patches.Circle((50, 100), avg_distance_to_goal, ec=point_color, fc='None', alpha=0.5, ls=':', lw=1.5)
    ax.add_patch(circle)
    ax.annotate("", xy=(50-avg_distance_to_goal, 101.5), xycoords='data', 
                xytext=(50, 101.5), textcoords='data',
                size=10, arrowprops=dict(arrowstyle="-|>",fc='none', ls='--', ec=player_col))
    ax.text(49-avg_distance_to_goal, 101.5, "{:.2f}m".format(avg_distance_to_goal), size=10, 
            ha='left', va='center',c=in_col)
    
    # Dynamic label based on whether all shots were goals
    label_text = "Avg goal distance" if total_shots == total_goals else "Avg shot distance"
    ax.text(51, 101.5, label_text, size=8, ha='right', va='center')

    # TEXT
    ax.scatter(np.linspace(90,10,5), [56]*5, c=bg_color, ec=point_color, marker='h', s=2000, ls='--', alpha=0.5)
    
    ax.text(90, 56,'{}'.format(total_shots), size=13, color=player_col, ha='center', va='center')
    ax.text(70, 56,'{}'.format(total_goals), size=13, color=player_col, ha='center', va='center')
    ax.text(50, 56,'{:.2f}'.format((total_goals-total_xg)/total_shots), size=13, color=player_col, 
        ha='center', va='center')
    ax.text(30, 56,'{:.2f}'.format(total_xg/total_shots), size=13, color=player_col, 
        ha='center', va='center')
    
    # Handle Changed Game State safely
    changed_gs_count = sum(df['Changed Game State']==True) if 'Changed Game State' in df.columns else 0
    ax.text(10, 56,'{}'.format(changed_gs_count), size=13, color=player_col, 
        ha='center', va='center')
    
    ax.text(90, 50,'Total\nShots', size=8, color=in_col, ha='center', va='top')
    ax.text(70, 50,'Total\nGoals', size=8, color=in_col, ha='center', va='top')
    ax.text(50, 50,'Finishing\nOverperformance\nper Shot', size=8, color=in_col, ha='center', va='top')
    ax.text(30, 50,'xG\nper Shot', size=8, color=in_col, ha='center', va='top')
    ax.text(10, 50,'Number of\nGame-State\nAltering Goals', size=8, color=in_col, ha='center', va='top')

    # ADD TITLES
    ax.set_title(f'{player.upper()}', color=gold_col, loc='left', 
        pad=21, weight='heavy', size=22)
    ax.text(103, 104,  f'{start_year}-{end_year+1} | League games only', size=13)
    fig.text(0.15, 0.02, 'Made using State of Play App. By Shivam Singh.', size=7)

    return fig, ax

def plot_pitch_with_zones(df, theme='light', player_col='red', alpha=0.5, show_zones=True):
    """
    Enhanced version with optional zone overlays
    
    Parameters:
    -----------
    show_zones : bool
        If True, adds semi-transparent zone boundaries to the pitch
    """
    if theme == 'light':
        point_color, bg_color = light_theme()
    elif theme == 'dark':
        point_color, bg_color = dark_theme()
    else:
        point_color, bg_color = light_theme()
        
    gold_col = '#ad993c'
    in_col = '#868686'

    fig, ax = plt.subplots(figsize=(10, 12))
    fig.set_facecolor(bg_color)
    pitch = mplsoccer.VerticalPitch(pitch_type='opta', pitch_color=bg_color,
                        linewidth=1.5, line_alpha=0.5, goal_alpha=0.5, half=True)
    pitch.draw(ax=ax)

    if len(df) == 0:
        return fig, ax

    # OPTIONAL: Draw zone boundaries
    if show_zones:
        zone_alpha = 0.08
        zone_color = point_color
        
        # Vertical zones (depth)
        ax.axhline(y=88, color=zone_color, linestyle=':', alpha=zone_alpha*2, linewidth=1)
        ax.axhline(y=83, color=zone_color, linestyle=':', alpha=zone_alpha*2, linewidth=1)
        ax.axhline(y=66, color=zone_color, linestyle=':', alpha=zone_alpha*2, linewidth=1)
        
        # Horizontal zones (width)
        ax.axvline(x=20, color=zone_color, linestyle=':', alpha=zone_alpha*2, linewidth=1)
        ax.axvline(x=40, color=zone_color, linestyle=':', alpha=zone_alpha*2, linewidth=1)
        ax.axvline(x=60, color=zone_color, linestyle=':', alpha=zone_alpha*2, linewidth=1)
        ax.axvline(x=80, color=zone_color, linestyle=':', alpha=zone_alpha*2, linewidth=1)
        
        # Add zone labels
        zone_label_size = 7
        zone_label_alpha = 0.3
        ax.text(50, 94, '6YB', size=zone_label_size, ha='center', va='center', 
                alpha=zone_label_alpha, color=in_col, style='italic')
        ax.text(50, 85.5, 'PA', size=zone_label_size, ha='center', va='center',
                alpha=zone_label_alpha, color=in_col, style='italic')
        ax.text(50, 74, 'Edge', size=zone_label_size, ha='center', va='center',
                alpha=zone_label_alpha, color=in_col, style='italic')

    # SCATTER POINTS with size variation by xG
    goal_cond = df['Shot Result']=='Goal'
    
    if len(df[~goal_cond]) > 0:
        sizes_non_goal = df[~goal_cond]['Shot xG'] * 300 + 50
        pitch.scatter(df[~goal_cond]['x']*100, df[~goal_cond]['y']*100, 
                     s=sizes_non_goal, c='none', ec=point_color, alpha=0.15, ax=ax)
    
    if len(df[goal_cond]) > 0:
        sizes_goal = df[goal_cond]['Shot xG'] * 400 + 100
        pitch.scatter(df[goal_cond]['x']*100, df[goal_cond]['y']*100, 
                     s=sizes_goal, c='none', ec=player_col, hatch='////////', alpha=alpha, ax=ax)

    # INTERNAL DATA
    player = df['Player'].iloc[0] if 'Player' in df.columns else "Unknown"
    start_year = df['Season'].min() if 'Season' in df.columns else 2024
    end_year = df['Season'].max() if 'Season' in df.columns else 2024
    avg_x, avg_y = df['x'].mean()*100, df['y'].mean()*100
    total_goals = sum(goal_cond)
    total_shots = len(df)
    total_xg = df['Shot xG'].sum()
    avg_distance_to_goal = np.sqrt(np.sum(np.square(np.array([100, 50])-
                                                    np.array([avg_x, avg_y]))))
    
    # DISTANCE CIRCLE
    # circle = patches.Circle((50, 100), avg_distance_to_goal, ec=point_color, fc='None', 
    #                        alpha=0.5, ls=':', lw=1.5)
    # ax.add_patch(circle)
    # ax.annotate("", xy=(50-avg_distance_to_goal, 101.5), xycoords='data', 
    #             xytext=(50, 101.5), textcoords='data',
    #             size=10, arrowprops=dict(arrowstyle="-|>",fc='none', ls='--', ec=player_col))
    # ax.text(49-avg_distance_to_goal, 101.5, "{:.2f}m".format(avg_distance_to_goal), size=10, 
    #         ha='left', va='center',c=in_col)
    # ax.text(51, 101.5, "Avg distance", size=8, ha='right', va='center')
    # DISTANCE CIRCLE
    circle = patches.Circle((50, 100), avg_distance_to_goal, ec=point_color, fc='None', 
                           alpha=0.5, ls=':', lw=1.5)
    ax.add_patch(circle)
    ax.annotate("", xy=(50-avg_distance_to_goal, 101.5), xycoords='data', 
                xytext=(50, 101.5), textcoords='data',
                size=10, arrowprops=dict(arrowstyle="-|>",fc='none', ls='--', ec=player_col))
    ax.text(49-avg_distance_to_goal, 101.5, "{:.2f}m".format(avg_distance_to_goal), size=10, 
            ha='left', va='center',c=in_col)
    
    # Dynamic label based on whether all shots were goals
    label_text = "Avg goal distance" if total_shots == total_goals else "Avg shot distance"
    ax.text(51, 101.5, label_text, size=8, ha='right', va='center')

    # STATS BOXES
    ax.scatter(np.linspace(90,10,5), [56]*5, c=bg_color, ec=point_color, marker='h', 
              s=2000, ls='--', alpha=0.5)
    
    ax.text(90, 56,'{}'.format(total_shots), size=13, color=player_col, ha='center', va='center')
    ax.text(70, 56,'{}'.format(total_goals), size=13, color=player_col, ha='center', va='center')
    ax.text(50, 56,'{:.2f}'.format((total_goals-total_xg)/total_shots), size=13, color=player_col, 
        ha='center', va='center')
    ax.text(30, 56,'{:.2f}'.format(total_xg/total_shots), size=13, color=player_col, 
        ha='center', va='center')
    
    changed_gs_count = sum(df['Changed Game State']==True) if 'Changed Game State' in df.columns else 0
    ax.text(10, 56,'{}'.format(changed_gs_count), size=13, color=player_col, 
        ha='center', va='center')
    
    ax.text(90, 50,'Total\nShots', size=8, color=in_col, ha='center', va='top')
    ax.text(70, 50,'Total\nGoals', size=8, color=in_col, ha='center', va='top')
    ax.text(50, 50,'Finishing\nOverperformance\nper Shot', size=8, color=in_col, ha='center', va='top')
    ax.text(30, 50,'xG\nper Shot', size=8, color=in_col, ha='center', va='top')
    ax.text(10, 50,'Number of\nGame-State\nAltering Goals', size=8, color=in_col, ha='center', va='top')

    # TITLES
    ax.set_title(f'{player.upper()}', color=gold_col, loc='left', 
        pad=21, weight='heavy', size=22)
    ax.text(103, 104,  f'{start_year}-{end_year+1} | League games only', size=13)
    fig.text(0.15, 0.02, 'Made using State of Play App. By Shivam Singh.', size=7)

    return fig, ax


def plot_heatmap(df, theme='light', cmap='hot', bins=15):
    """
    Create a hexbin heatmap of shot locations
    
    Parameters:
    -----------
    df : DataFrame
        Shot data with x, y coordinates
    theme : str
        'light' or 'dark'
    cmap : str
        Matplotlib colormap name
    bins : int
        Number of hexagonal bins
    """
    if theme == 'light':
        point_color, bg_color = light_theme()
    elif theme == 'dark':
        point_color, bg_color = dark_theme()
    else:
        point_color, bg_color = light_theme()
    
    fig, ax = plt.subplots(figsize=(10, 12))
    fig.set_facecolor(bg_color)
    pitch = mplsoccer.VerticalPitch(pitch_type='opta', pitch_color=bg_color,
                        linewidth=1.5, line_alpha=0.5, goal_alpha=0.5, half=True)
    pitch.draw(ax=ax)
    
    if len(df) == 0:
        return fig, ax
    
    # Create hexbin heatmap
    hexbin = pitch.hexbin(df['x']*100, df['y']*100, ax=ax, 
                         edgecolors=bg_color, gridsize=bins,
                         cmap=cmap, alpha=0.7, mincnt=1)
    
    # Add colorbar
    cbar = plt.colorbar(hexbin, ax=ax, pad=0.03)
    cbar.set_label('Shot Frequency', color=point_color)
    cbar.ax.yaxis.set_tick_params(color=point_color)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=point_color)
    
    # Add title
    player = df['Player'].iloc[0] if 'Player' in df.columns else "Unknown"
    start_year = df['Season'].min() if 'Season' in df.columns else 2024
    end_year = df['Season'].max() if 'Season' in df.columns else 2024
    ax.set_title(f'{player.upper()} - SHOT HEATMAP', 
                color='#ad993c', loc='left', pad=21, weight='heavy', size=22)
    ax.text(103, 104, f'{start_year}-{end_year+1} | League games only', size=13)
    
    return fig, ax


def create_enhanced_visualization(df, viz_type='standard', **kwargs):
    """
    Smart wrapper that creates different visualization types
    
    Parameters:
    -----------
    viz_type : str
        'standard' - Original shot map
        'zones' - Shot map with zone overlays
        'heatmap' - Hexbin heatmap
    **kwargs : additional arguments passed to plotting function
    """
    viz_functions = {
        'standard': plot_pitch,
        'zones': plot_pitch_with_zones,
        'heatmap': plot_heatmap
    }
    
    if viz_type not in viz_functions:
        print(f"Unknown viz_type '{viz_type}'. Using 'standard'.")
        viz_type = 'standard'
    
    return viz_functions[viz_type](df, **kwargs)





