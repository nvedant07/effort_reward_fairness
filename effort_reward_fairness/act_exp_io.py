import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pandas as pd
import sys, os, time
import operator
import shelve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from sklearn.externals import joblib
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydotplus

import cost_funcs as cf
import group_explanations as ge

sys.path.append('../util/')
from datasets import file_util as fu
import output as out

INF = 1000000000000
colors = ['red', 'blue', 'orange', 'green', 'maroon', 'magenta', 'royalblue', 'yellow', 'cyan', 'indigo', 'goldenrod', 'lightpink', 'black']
# This file contains IO helper functions

def get_group_descr(group, feature_infos, add_cols=[], add_col_labels=[], wiki_format=True, add_col_formats=None, transpose=False):
    """Prints a table showing the feature values for a group of users

    Keyword arguments:
    group -- the group of users
    feature_info -- a list with information about names, types and values
        of all the features in the dataset.
    add_cols -- additional columns to append to the table, e.g. for showing predicted labels or efforts (default [])
    add_col_labels -- the names of the additional columns (default [])
    wiki_format -- whether to print the table formatted for wiki or console output (default True)
    add_col_format -- format specifies for additional columns, e.g. for showing them as percentages or limiting the number of digits, can be used if wiki_format = True
    tranpose -- whether to transpose the table, useful for displaying many features with console output (default False)
    """

    conv_group = []
    for user in group:
        _, conv_user = cf.bin_to_index_vals(feature_infos, user, cont_permitted=True)
        conv_group.append(conv_user)

    #col_names = [fname + '_' + flabels[0][1] if ftype == fu.ATTR_CAT_BIN else fname for fname, ftype, flabels in feature_infos]
    col_names = [fname for fname, _, _ in feature_infos]

    cols = [pd.DataFrame(np.array(col).T, columns=[col_label]) for col, col_label in zip(add_cols, add_col_labels)]

    df = pd.DataFrame(conv_group, columns=col_names)
    df = pd.concat([df] + cols, axis=1)
    if transpose:
        df = df.T

    if wiki_format:
        full_col_names = ["User"] + df.columns.values.tolist()
        values = df.values
        values = np.c_[np.arange(len(values)), values]
        if add_col_formats is not None:
            add_col_formats = [''] * (len(col_names) + 1) + add_col_formats
        return out.get_table(full_col_names, values, val_format=add_col_formats)
    else:
        return str(df)

def print_explanation(feature_infos, var_indices, var_values):
    """Prints explanations to the console"""
    print("\nRequired assignment:\n", get_conditions_str(feature_infos, var_indices, var_values))

def get_conditions_str(feature_infos, nec_vars, vals, scaler=None, level=0):
    """Constructs a printable version of an explanation given as feature indices and the values to set them to"""
    conditions = ge.get_conditions(feature_infos, nec_vars, vals)
    if scaler is not None:
        ranges = scaler.data_range_
        minimums = scaler.data_min_
        for k,v in conditions.items():
            original_val = v[1] * ranges[v[0]] + minimums[v[0]]
            conditions[k] = (v[0], original_val)
    cond_list = ["{} = {}".format(feature, str(val)) for feature, (_, val) in conditions.items()]
    return out.get_listing(cond_list, level=level)

SERVER_PROJECT_PATH = 'effort_reward_fairness'

def get_wiki_link(figure_path):
    """Returns the link to an image in wiki format"""
    server_path = figure_path.replace('results', SERVER_PROJECT_PATH)
    wiki_link = out.web_attachment(server_path, size=700)
    return wiki_link

def plot_effort_histogram(res_dir, gname, efforts):
    """Computes a histogram of efforts

    Keyword arguments:
    res_dir -- the directory to store the histogram in
    gname -- the name of the group
    efforts -- the list of efforts for the users in the group
    """

    fig = plt.figure()
    ax = fig.gca()

    n, bins, patches = ax.hist(efforts)
    ax.set_title("Group {}".format(gname))
    ax.set_xlabel("Efforts")
    ax.set_ylabel("Number of users")

    #plt.show()
    figpath = res_dir + "/" + gname + "_efforts.png"
    fig.savefig(figpath)
    return get_wiki_link(figpath)

def get_dict_listing(res_dict, level=0, sort_by=None):
    """Converts a dictionary to a wiki listing"""
    assert sort_by is None or sort_by in ['key', 'val']

    if sort_by is None:
        itemgetter = res_dict.items()
    else:
        ind = 0 if sort_by == 'key' else 1
        itemgetter = sorted(res_dict.items(), key=lambda p: p[ind])

    stats_str = out.get_listing(["{}: {:.3f}".format(
        stat_name, stat) for stat_name, stat in itemgetter],
        level=level)
    return stats_str

def print_rule_stats(res_file, targeting_cnf, performance_stats):
    """Writes statistics about a targeting rule to a file"""
    res_file.write("== Targeting rule ==\n\n")
    res_file.write("Full Rule:\n{}\n\n".format(targeting_cnf.to_string(wiki_linebreaks=True)))
    res_file.write("Cleaned up Rule:\n{}\n\n".format(targeting_cnf.to_consolidated_string(wiki_linebreaks=True)))
    
    stats_str = get_dict_listing(performance_stats)
    res_file.write(stats_str + "\n")

def get_effort_stats(res_dir, gname, efforts):
    """Computes effort statistics"""
    effort_stats = {"Average": np.mean(efforts),
            "Mininum": np.min(efforts),
            "Maximum": np.max(efforts)}
    histogram_link = plot_effort_histogram(res_dir, gname, efforts)
    effort_stats_str = get_dict_listing(effort_stats) + "\n{}\n".format(histogram_link)
    return effort_stats_str

def get_exp_key(nec_vars, nec_vals):
   sorted_vals = tuple(val for _, val in sorted(zip(nec_vars, nec_vals), key=lambda t: t[0]))
   sorted_vars = tuple(sorted(nec_vars))
   return (sorted_vars, sorted_vals)

def plot_explanation_hist(res_dir, sorted_explanations):
    """Plots a histogram showing the occurrence frequencies of
    different explanations"""
    explanation_counts = [c for _, c in sorted_explanations]

    fig = plt.figure()
    ax = fig.gca()

    ax.bar(np.arange(len(explanation_counts)), explanation_counts)
    ax.set_title("Explanation occurences")
    ax.set_xlabel("Explanations")
    ax.set_ylabel("Number of receiving users")

    #plt.show()
    figpath = res_dir + "/exp_occurrences.png"
    fig.savefig(figpath)
    return get_wiki_link(figpath)

def get_column_heading_order(row):
    row = list(row)
    row = [float(val.split('(')[0].strip()) for val in row]
    ordering, _ = list(zip(*sorted(enumerate(row), key=operator.itemgetter(1), reverse=True)))
    return list(ordering)

def get_disparity_plots(res_dir, col_headings, values, plot_title='', filename='all_disp_in_one',format='png'):
    plt.rcParams['font.size'] = 24
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['axes.labelsize'] = 22
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['axes.linewidth'] = 3
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['figure.titlesize'] = 28
    plt.rcParams['lines.linewidth'] = 3.0

    plots_dir = res_dir + "/disparity_plots"
    out.create_dir(plots_dir)
    values = np.array(values)
    x_labels = list(values[:,0])
    x_labels = [label.split("<<BR>>")[0].strip() for label in x_labels]
    try:
        x_labels[x_labels.index("LogReg")] = "Log\nReg"
        x_labels[x_labels.index("NeuralNet")] = "Neural\nNet"
    except:
        pass
    x_title, y_title = "Model", "Disparity"
    wiki_string = ""
    width = 0.2
    x_vals = np.array(range(1, 2*len(x_labels), 2))

    order_of_cols = get_column_heading_order(values[0,1:].flatten())
    values[:,1:] = values[:,1:][:,order_of_cols]
    col_headings = np.array(col_headings)
    print (col_headings, type(col_headings), order_of_cols)
    col_headings[1:] = col_headings[1:][order_of_cols]
    col_headings = list(col_headings)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    all_rects, all_disp_types = [], []

    for i in range(1,len(col_headings)):
        if 'statistical' in col_headings[i].lower():
            disparity_type = "Statistical\nDisparity"
        else:
            disparity_type = col_headings[i].split("<<BR>>")[0].replace('Disparity ','Disparity\n').replace(' (','\n(')
        y_vals = list(values[:,i].flatten())
        y_vals = ([float(val.split('(')[0].strip()) for val in y_vals])    
        rect = ax.bar(x_vals + (i-1)*width, y_vals, width, color=colors[i-1])
        all_rects.append(rect)
        all_disp_types.append(disparity_type)
    ax.set_title(plot_title)
    ax.set_yscale('log')
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    ax.set_xticks(x_vals + (len(col_headings) - 2) * width/2)
    ax.set_xticklabels(x_labels)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(all_rects, all_disp_types, loc='center left', bbox_to_anchor=(1, 0.5))
    figpath = plots_dir + '/' + filename + '.' + format
    plt.savefig(figpath, format=format, bbox_inches='tight')
    plt.savefig(plots_dir + '/' + filename + '.pdf', format='pdf', bbox_inches='tight')
    plt.clf()
    plt.close()
    wiki_string += "\n{}\n\n".format(get_wiki_link(figpath))
    return wiki_string

def get_utility_threshold_plots(res_dir, models, col_headings, values, values_old, tau_nosens, plot_title='', filename='utility_threshold', format='png'):
    plt.rcParams['font.size'] = 24
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['figure.titlesize'] = 28
    plt.rcParams['lines.linewidth'] = 1.0

    plots_dir = res_dir + "/disparity_plots"
    out.create_dir(plots_dir)
    values = np.array(values)
    x_title, y_title = r'$C_s$', col_headings[2].split("<<BR>>")[0].strip()
    # plot_title = r'$C$_~s = ' + str(tau_nosens)
    plot_title = ''

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    all_model_names = [str(x) for x in models]
    for i in range(len(all_model_names)):
        mask = np.where(values[:,0] == all_model_names[i])[0]
        x_labels, y_vals = values[mask,1].flatten(), list(values[mask,2].flatten())
        y_vals_old = list(values_old[mask,2].flatten())
        y_vals_old = [float(val.split('(')[0].strip()) for val in y_vals_old]
        x_labels = ["{:.2f}".format(float(x)) for x in x_labels]
        x_vals = np.arange(1,len(x_labels)+1, 1)
        y_vals = [float(val.split('(')[0].strip()) for val in y_vals]
        ax.plot(x_vals, y_vals, color=colors[i], marker='o', label=all_model_names[i].split("<<BR>>")[0])
        ax.plot(x_vals, y_vals_old, color=colors[i], linestyle='dashed', marker='o')
    ax.set_title(plot_title)
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    ax.set_xticks(x_vals)
    ax.set_xticklabels(x_labels)
    # if 'fpr' in y_title.lower():
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    figpath = plots_dir + '/' + filename + '.' + format
    plt.savefig(figpath, format=format, bbox_inches='tight')
    plt.savefig(plots_dir + '/' + filename + '.pdf', format='pdf', bbox_inches='tight')
    for i, ax_obj in enumerate([ax]):
        fig_legend = plt.figure(figsize=(3, 3))
        handles, labels = ax_obj.get_legend_handles_labels()
        fig_legend.legend(handles, labels, 'center', ncol=1)
        fig_legend.savefig(plots_dir + '/' + filename + "_legend." + format, format=format, bbox_inches='tight')
        fig_legend.savefig(plots_dir + '/' + filename + "_legend.pdf", format='pdf', bbox_inches='tight')
    plt.clf()
    plt.close()
    wiki_string = "\n{}\n\n{}\n\n".format(get_wiki_link(figpath), get_wiki_link(plots_dir + '/' + filename + "_legend." + format))
    return wiki_string

def get_abs_clustering_plots(res_dir, thresholds, new_abs_clustering_index, old_abs_clustering_index, 
    tau_nosens, y_title, plot_title='', filename='abs_index_utility_threshold', format='png'):
    plt.rcParams['font.size'] = 24
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['axes.labelsize'] = 22
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['axes.linewidth'] = 3
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['figure.titlesize'] = 28
    plt.rcParams['lines.linewidth'] = 3.0

    plots_dir = res_dir + "/disparity_plots"
    out.create_dir(plots_dir)
    x_title, plot_title = r'$C_s$', r'$C$_~s = ' + str(tau_nosens)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    count, min_y, max_y = 0, INF, -INF
    for k, v in new_abs_clustering_index.items():
        y_vals, y_vals_old = v, [old_abs_clustering_index[k]] * len(thresholds) if old_abs_clustering_index is not None else None
        x_labels = ["{:.2f}".format(float(x)) if float(10000*x)%100 == 0 else '' for x in thresholds]
        x_vals = np.arange(1,len(x_labels)+1, 1)
        ax.plot(x_vals, y_vals, color=colors[count], marker='o', label=k.filename())
        if y_vals_old is not None:
            ax.plot(x_vals, y_vals_old, color=colors[count], linestyle='dashed', marker='o')
        count += 1
    # ax.set_ylim(min_y - 0.5, max_y + 0.5)
    # ax.set_yticks(list(ax.get_yticks())[:-1])
    ax.set_title(plot_title)
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    ax.set_xticks(x_vals)
    ax.set_xticklabels(x_labels)
    # if 'fpr' in y_title.lower():
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    figpath = plots_dir + '/' + filename + '.' + format
    plt.savefig(figpath, format=format, bbox_inches='tight')
    plt.savefig(plots_dir + '/' + filename + '.pdf', format='pdf', bbox_inches='tight')
    # for i, ax_obj in enumerate([ax]):
    #     fig_legend = plt.figure(figsize=(4, 3))
    #     handles, labels = ax_obj.get_legend_handles_labels()
    #     fig_legend.legend(handles, labels, 'center', ncol=1)
    #     fig_legend.savefig(plots_dir + '/' + filename + "_legend." + format, format=format, bbox_inches='tight')
    #     fig_legend.savefig(plots_dir + '/' + filename + "_legend.pdf", format='pdf', bbox_inches='tight')
    plt.clf()
    plt.close()
    wiki_string = "\n{}\n\n".format(get_wiki_link(figpath))
    return wiki_string

def get_pdf_plots(res_dir, X, sens_group, taus_for_sens, feature_info, data_for_pdf, tau_nosens, separate_legend=False, y_title='', 
        combine_all_plots=False, plot_title='', filename='', format='png'):
    assert len(feature_info) == X.shape[1]
    plt.rcParams['font.size'] = 24
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['axes.labelsize'] = 22
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['axes.linewidth'] = 3
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['figure.titlesize'] = 28
    plt.rcParams['lines.linewidth'] = 3.0

    plots_dir = res_dir + "/pdf_before_after_plots"
    out.create_dir(plots_dir)

    X_sens, X_nosens = X[sens_group], X[~sens_group]

    for i in range(len(feature_info)):
        for model, tau_sens_to_population in data_for_pdf.items():
            plot_title = "{}, feature name: {} ({}), C_~s = {}".format(model.filename(), feature_info[i][0], feature_info[i][1], tau_nosens)
            filename = "pdf_{}_{}_{}".format(model.filename(), tau_nosens, feature_info[i][0])
            figpath = plots_dir + '/' + filename + '.' + format
            figpath_full = plots_dir + '/' + filename + '_complete_population.' + format
            fig1, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(6, 6))
            fig3 = plt.figure(figsize=(4, 4))
            ax3 = fig3.add_subplot(111)
            fig1.suptitle(plot_title)
            ax1.set_title("Sens Group")
            ax2.set_title("Non-Sens Group")
            x_vals = sorted(list(set(X[:,i])))
            y_vals_sens = [np.count_nonzero(X_sens[:,i] == x_val)/len(X_sens) for x_val in x_vals]
            y_vals_nosens = [np.count_nonzero(X_nosens[:,i] == x_val)/len(X_nosens) for x_val in x_vals]
            y_vals_all = [np.count_nonzero(X[:,i] == x_val)/len(X) for x_val in x_vals]
            ax1.plot(x_vals, y_vals_sens, color=colors[0], marker='', linestyle='dashed')
            ax2.plot(x_vals, y_vals_nosens, color=colors[0], marker='', linestyle='dashed', label='Original Population')
            ax3.plot(x_vals, y_vals_all, color=colors[0], marker='', linestyle='dashed', label='Original Population')
            for tau_sens, population in tau_sens_to_population.items():
                idx = taus_for_sens.index(tau_sens)
                population_sens, population_nosens = population[sens_group], population[~sens_group]
                y_vals_sens = [np.count_nonzero(population_sens[:,i] == x_val)/len(population_sens) for x_val in x_vals]
                y_vals_nosens = [np.count_nonzero(population_nosens[:,i] == x_val)/len(population_nosens) for x_val in x_vals]
                y_vals_all = [np.count_nonzero(population[:,i] == x_val)/len(population) for x_val in x_vals]
                ax1.plot(x_vals, y_vals_sens, color=colors[idx + 1], marker='')
                ax2.plot(x_vals, y_vals_nosens, color=colors[idx + 1], marker='', label='{}'.format(tau_sens))
                ax3.plot(x_vals, y_vals_all, color=colors[idx + 1], marker='', label='{}'.format(tau_sens))
            if not separate_legend:
                box2, box3 = ax2.get_position(), ax3.get_position()
                ax2.set_position([box2.x0, box2.y0, box2.width * 0.8, box2.height])
                ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                ax3.set_position([box3.x0, box3.y0, box3.width * 0.8, box3.height])
                ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            fig1.savefig(figpath, format=format, bbox_inches='tight')
            fig1.savefig(plots_dir + '/' + filename + '.pdf', format='pdf', bbox_inches='tight')
            fig3.savefig(figpath_full, format=format, bbox_inches='tight')
            fig3.savefig(plots_dir + '/' + filename + '_complete_population.pdf', format='pdf', bbox_inches='tight')
            # if separate_legend:
            #     fig_legend = plt.figure(figsize=(4, 3))
            #     handles, labels = ax2.get_legend_handles_labels()
            #     fig_legend.legend(handles, labels, 'center', ncol=1)
            #     fig_legend.savefig(plots_dir + '/' + filename + "_legend" + format, format=format, bbox_inches='tight')
            #     fig_legend.savefig(plots_dir + '/' + filename + "_legend.pdf", format='pdf', bbox_inches='tight')
            plt.close(fig1)
            plt.close(fig3)
            wiki_string = "\n{}\n\n{}\n\n".format(get_wiki_link(figpath), get_wiki_link(figpath_full))
            yield wiki_string

def get_segregation_plots(res_dir, outer_seg_index_mapping, format='png'):
    plt.rcParams['font.size'] = 24
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['axes.labelsize'] = 22
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['axes.linewidth'] = 3
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['figure.titlesize'] = 28
    plt.rcParams['lines.linewidth'] = 3.0

    plots_dir = res_dir + "/segregation_plots"
    out.create_dir(plots_dir)
    x_title, y_title = r'$C_s$', r'$C$_~s'

    threed_plot_mapping = {} # mapping from { SSI : {model: ([<x_vals>], [<y_vals>], [<z_vals>])} }
    plot_strings_mapping = {} # mapping from { SSI : [<str1>, <str2>....] }
    strings_to_write = []

    print ("Initially passed dict: {}".format(outer_seg_index_mapping))

    for tau_nosens, seg_index_mapping in outer_seg_index_mapping.items():
        plot_strings_mapping[tau_nosens] = {}
        for index_type, mapping in seg_index_mapping.items():
            if index_type.shortname() not in threed_plot_mapping:
                threed_plot_mapping[index_type.shortname()] = {}
            # plot_title = str(index_type)
            plot_title = ''
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111)
            ax.get_yaxis().get_major_formatter().set_useOffset(False)
            # ax.set_yscale('log')
            count, min_y, max_y = 0, INF, -INF
            for model, tau_mapping in mapping.items():
                y_vals_old = tau_mapping.pop('Original Population')
                x_labels, y_vals = list(zip(*tau_mapping.items()))
                x_labels = list(map(float, x_labels))
                x_labels, y_vals = list(zip(*(sorted(zip(x_labels, y_vals), key=operator.itemgetter(0)))))
                y_vals_old = [y_vals_old] * len(x_labels) if y_vals_old is not None else None
                x_vals = np.arange(1,len(x_labels)+1, 1)
                ax.plot(x_vals, y_vals, color=colors[count], marker='o', label=model.filename())
                if y_vals_old is not None:
                    ax.plot(x_vals, y_vals_old, color=colors[-1], linestyle='dashed', marker='o', label='Original Population')

                if model in threed_plot_mapping[index_type.shortname()]:
                    threed_plot_mapping[index_type.shortname()][model][0] += list(map(float, x_labels))
                    threed_plot_mapping[index_type.shortname()][model][1] += [tau_nosens] * len(x_labels)
                    threed_plot_mapping[index_type.shortname()][model][2] += list(y_vals) # X, Y, Z
                    threed_plot_mapping[index_type.shortname()]['Original Population'][0] += list(map(float, x_labels))
                    threed_plot_mapping[index_type.shortname()]['Original Population'][1] += [tau_nosens] * len(x_labels)
                    threed_plot_mapping[index_type.shortname()]['Original Population'][2] += list(y_vals_old)
                else:
                    threed_plot_mapping[index_type.shortname()][model] = [list(map(float, x_labels)), [tau_nosens] * len(x_labels), list(y_vals)] # X, Y, Z
                    threed_plot_mapping[index_type.shortname()]['Original Population'] = [list(map(float, x_labels)), [tau_nosens] * len(x_labels), list(y_vals_old)] # X, Y, Z

                count += 1
            ax.set_title(plot_title)
            ax.set_ylabel(str(index_type))
            ax.set_xlabel(x_title)
            ax.set_xticks(x_vals)
            ax.set_xticklabels(x_labels)
            # box = ax.get_position()
            # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            
            filename = "segregation_{}_{}".format(index_type.shortname(), tau_nosens)
            figpath = plots_dir + '/' + filename + '.' + format
            plt.savefig(figpath, format=format, bbox_inches='tight')
            plt.savefig(plots_dir + '/' + filename + '.pdf', format='pdf', bbox_inches='tight')
            for i, ax_obj in enumerate([ax]):
                fig_legend = plt.figure(figsize=(4, 3))
                handles, labels = ax_obj.get_legend_handles_labels()
                fig_legend.legend(handles, labels, 'center', ncol=1)
                fig_legend.savefig(plots_dir + '/' + filename + "_legend." + format, format=format, bbox_inches='tight')
                fig_legend.savefig(plots_dir + '/' + filename + "_legend.pdf", format='pdf', bbox_inches='tight')
            plt.close(fig)

            if index_type.shortname() in plot_strings_mapping[tau_nosens]:
                plot_strings_mapping[tau_nosens][index_type.shortname()].append("{}\n\n{}".format(get_wiki_link(figpath), get_wiki_link(plots_dir + '/' + filename + "_legend." + format)))
            else:
                plot_strings_mapping[tau_nosens][index_type.shortname()] = ["{}\n\n{}".format(get_wiki_link(figpath), get_wiki_link(plots_dir + '/' + filename + "_legend." + format))]

    print ("3D plot mapping: {}".format(threed_plot_mapping))
    print ("Plot string mapping: {}".format(plot_strings_mapping))

    for seg_index, model_mapping in threed_plot_mapping.items():
        strings_to_write.append("== {} ==".format(str(seg_index)))

        for tau_nosens, seg_index_mapping in plot_strings_mapping.items():
            strings_to_write.append("=== Tau for non-sens: {} ===".format(tau_nosens))
            strings_to_write += seg_index_mapping[seg_index]

        # plot_title = str(seg_index)
        plot_title = ''
        filename = "segregation_{}_3d".format(seg_index)
        figpath = plots_dir + '/' + filename + '.' + format
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        count = 0
        for model, x_y_z in model_mapping.items():
            x_meshgrid, y_meshgrid = np.meshgrid(x_y_z[0], x_y_z[1])
            _, z_meshgrid = np.meshgrid(x_y_z[0], x_y_z[2])
            ax.plot_wireframe(x_meshgrid, y_meshgrid, z_meshgrid, label=model if isinstance(model, str) else model.filename(), 
                linestyle='dashed' if isinstance(model, str) else 'solid', color=colors[count])
            count += 1
        ax.set_title(plot_title)
        ax.set_xlabel(x_title)
        ax.set_ylabel(y_title)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(figpath, format=format, bbox_inches='tight')
        plt.savefig(plots_dir + '/' + filename + '.pdf', format='pdf', bbox_inches='tight')
        plt.close(fig)
        strings_to_write.append(get_wiki_link(figpath))

    return strings_to_write

def get_segregation_plots_new(res_dir, outer_seg_index_mapping, fc, format='png'):
    """
    This function was written very very close to the deadline.
    """
    plt.rcParams['font.size'] = 24
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['axes.labelsize'] = 22
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['axes.linewidth'] = 3
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 28
    plt.rcParams['lines.linewidth'] = 3.0

    plots_dir = res_dir + "/segregation_plots"
    out.create_dir(plots_dir)
    x_title = r'$\tau$'

    print ("Initially passed dict: {}".format(outer_seg_index_mapping))

    for tau_nosens, seg_index_mapping in outer_seg_index_mapping.items():
        if tau_nosens > 0:
            continue
        count = 3
        if fc:
            for index_type, mapping in seg_index_mapping.items():
                # plot_title = str(index_type)
                plot_title = ''
                fig = plt.figure(figsize=(4, 4))
                ax = fig.add_subplot(111)
                ax.get_yaxis().get_major_formatter().set_useOffset(False)
                x_vals, y_vals, y_vals_old = [], [], []
                for model, tau_mapping in mapping.items():
                    if '0.00' in tau_mapping:
                        x_vals.append(model.tau)
                        y_vals.append(tau_mapping.pop('0.00'))
                        for inner_model, inner_tau_mapping in mapping.items():
                            if 'Original Population' in inner_tau_mapping and model.tau == inner_model.tau:
                                y_vals_old.append(inner_tau_mapping['Original Population'])

                print (x_vals, y_vals_old, y_vals)
                x_vals, y_vals, y_vals_old = list(zip(*(sorted(zip(x_vals, y_vals, y_vals_old), key=operator.itemgetter(0)))))

                ax.plot(x_vals, y_vals, color=colors[count], marker='o', label='Impacted Population')
                ax.plot(x_vals, y_vals_old, color=colors[count + 1], marker='o', label='Initial Population')

                ax.set_title(plot_title)
                ax.set_ylabel(str(index_type))
                ax.set_xlabel(x_title)
                ax.set_xticks(x_vals)
                ax.set_xticklabels(list(map(str, x_vals)))
                # box = ax.get_position()
                # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                if 'atkinson' in index_type.shortname().lower():
                    ax.legend(loc='lower left')
                elif 'aci' in index_type.shortname().lower():
                    ax.legend(loc='upper right')
                else:
                    ax.legend(loc='lower right')
                filename = "segregation_{}_fc".format(index_type.shortname())
                figpath = plots_dir + '/' + filename + '.' + format
                plt.savefig(figpath, format=format, bbox_inches='tight')
                plt.savefig(plots_dir + '/' + filename + '.pdf', format='pdf', bbox_inches='tight')
                # for i, ax_obj in enumerate([ax]):
                #     fig_legend = plt.figure(figsize=(4, 3))
                #     handles, labels = ax_obj.get_legend_handles_labels()
                #     fig_legend.legend(handles, labels, 'center', ncol=1)
                #     fig_legend.savefig(plots_dir + '/' + filename + "_legend." + format, format=format, bbox_inches='tight')
                #     fig_legend.savefig(plots_dir + '/' + filename + "_legend.pdf", format='pdf', bbox_inches='tight')
                plt.close(fig)

                yield get_wiki_link(figpath)
        else:
            x_labels, y_old, y_new = [], [], []
            for index_type, mapping in seg_index_mapping.items():
                x_labels, y_old, y_new = [], [], []
                # plot_title = str(index_type)
                for model, tau_mapping in mapping.items():
                    if '0.00' in tau_mapping:
                        print ("Outer: {}".format(model.filename()))
                        # if index_type.shortname() == 'centralization':
                        #     x_labels.append('{}\n{}'.format(model.shortfilename(), 'Cent.'))
                        # elif index_type.shortname() == 'atkinson':
                        #     x_labels.append('{}\n{}'.format(model.shortfilename(), 'Atkinson'))
                        # else:
                        #     x_labels.append('{}\n{}'.format(model.shortfilename(), index_type.shortname()))
                        x_labels.append(model.shortfilename())
                        y_new.append(tau_mapping['0.00'])
                        for inner_model, inner_tau_mapping in mapping.items():
                            if 'Original Population' in inner_tau_mapping and model.filename() == inner_model.filename():
                                print ("Added")
                                y_old.append(inner_tau_mapping['Original Population'])
                                break
                
                fig = plt.figure(figsize=(4, 4))
                ax = fig.add_subplot(111)
                ax.get_yaxis().get_major_formatter().set_useOffset(False)
                ind = np.arange(len(y_old))
                width = 0.25

                min_y = min(min(y_new),min(y_old))
                max_y = max(max(y_new),max(y_old))
                range_y = max_y - min_y
                ax.set_ylim(min_y - range_y, max_y + range_y)

                ax.bar(ind + width, y_new, width=width, color=colors[0], label='Impacted Population')
                ax.bar(ind, y_old, width=width, color=colors[1], label='Initial Population')
                ax.set_xticks(ind + width/2)
                ax.set_ylabel(str(index_type))
                ax.set_xticklabels(x_labels)
                ax.legend(loc='upper right')
                filename = "segregation_{}".format(index_type.shortname())
                figpath = plots_dir + '/' + filename + '.' + format
                plt.savefig(figpath, format=format, bbox_inches='tight')
                plt.savefig(plots_dir + '/' + filename + '.pdf', format='pdf', bbox_inches='tight')
                plt.close(fig)
                yield get_wiki_link(figpath)

                # x_labels.append(' ')
                # y_old.append(0)
                # y_new.append(0)
            # print (x_labels)
            # print (y_old, y_new)
            # assert len(y_old) == len(y_new)
            # fig = plt.figure(figsize=(10, 4))
            # ax = fig.add_subplot(111)
            # ax.get_yaxis().get_major_formatter().set_useOffset(False)
            # # ax.set_yscale('log')

            # ind = np.arange(len(y_old))
            # width = 0.25
            # ax.bar(ind + width, y_new, width=width, color=colors[0], label='New Value')
            # ax.bar(ind, y_old, width=width, color=colors[1], label='Initial Value')
            # ax.set_xticks(ind)
            # ax.set_xticklabels(x_labels)
            
            # ax.legend(loc='upper right')

            # filename = "segregation_all_in_one"
            # figpath = plots_dir + '/' + filename + '.' + format
            # plt.savefig(figpath, format=format, bbox_inches='tight')
            # plt.savefig(plots_dir + '/' + filename + '.pdf', format='pdf', bbox_inches='tight')
            # plt.close(fig)

            # yield get_wiki_link(figpath)


def plot_covar_matrix(res_dir, X, feature_info, format='png'):
    plt.rcParams['font.size'] = 12
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 16
    plt.rcParams['figure.titlesize'] = 15
    plt.rcParams['lines.linewidth'] = 1.0

    covar_mat = np.cov(X, rowvar=False)
    assert covar_mat.shape[0] == X.shape[1] and covar_mat.shape[1] == X.shape[1]
    plots_dir = res_dir + "/disparity_plots"
    out.create_dir(plots_dir)
    filename = 'training_set_feature_covar_mat'
    figpath = plots_dir + '/' + filename + '.' + format
    plt.figure(figsize = (20,18))
    import seaborn as sns
    ax = sns.heatmap(covar_mat, annot=True, xticklabels=get_feature_names(feature_info), 
        yticklabels=get_feature_names(feature_info))
    ax.xaxis.tick_top() # x axis on top
    ax.xaxis.set_label_position('top')
    ax.tick_params(length=0)
    plt.xticks(rotation=1)
    plt.title('Covariance Matrix')
    plt.savefig(figpath, format=format, bbox_inches='tight')
    plt.savefig(plots_dir + '/' + filename + '.pdf', format='pdf', bbox_inches='tight')
    return get_wiki_link(figpath)

def plot_dtree(res_dir, clf, feature_info):
    plots_dir = res_dir + "/disparity_plots"
    out.create_dir(plots_dir)
    filename = "{}_viz".format(clf.filename())
    figpath = plots_dir + '/' + filename + '.png'
    dot_data = StringIO()
    export_graphviz(clf.clf, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True,
                    feature_names=get_feature_names(feature_info))
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(figpath)
    return get_wiki_link(figpath)

def plot_one_var_vs_other(res_dir, model, x_vals, y_vals_sens, y_vals_nosens, x_label, y_label, format='png'):
    plt.rcParams['font.size'] = 24
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['axes.labelsize'] = 22
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['axes.linewidth'] = 3
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['figure.titlesize'] = 28
    plt.rcParams['lines.linewidth'] = 3.0

    plots_dir = res_dir + "/disparity_plots"
    out.create_dir(plots_dir)
    filename = '{}_{}_vs_{}'.format(model.filename(), '_'.join(x_label.split()), '_'.join(y_label.split()))
    figpath = plots_dir + '/' + filename + '.' + format
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    ax.plot(x_vals, y_vals_sens, color='orange', label='Women')
    ax.plot(x_vals, y_vals_nosens, color='green', label='Men')
    ax.set_xlabel(x_label + ' (' + r'$\delta$' + ')')
    ax.set_ylabel(y_label)
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax.legend(loc='lower right', bbox_to_anchor=(1, 0.5))
    ax.legend(loc='lower right')
    plt.savefig(figpath, format=format, bbox_inches='tight')
    plt.savefig(plots_dir + '/' + filename + '.pdf', format='pdf', bbox_inches='tight')

    # for i, ax_obj in enumerate([ax]):
    #     fig_legend = plt.figure(figsize=(3, 3))
    #     handles, labels = ax_obj.get_legend_handles_labels()
    #     fig_legend.legend(handles, labels, 'center', ncol=1)
    #     fig_legend.savefig(plots_dir + '/' + filename + "_legend." + format, format=format, bbox_inches='tight')
    #     fig_legend.savefig(plots_dir + '/' + filename + "_legend.pdf", format='pdf', bbox_inches='tight')

    return "{}\n\n{}".format(get_wiki_link(figpath), get_wiki_link(plots_dir + '/' + filename + "_legend." + format))

def plot_one_var_vs_other_together(res_dir, sens_dict, nosens_dict, x_label, y_label, format='png'):
    plt.rcParams['font.size'] = 24
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['axes.labelsize'] = 22
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['axes.linewidth'] = 3
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['figure.titlesize'] = 28
    plt.rcParams['lines.linewidth'] = 3.0

    plots_dir = res_dir + "/disparity_plots"
    out.create_dir(plots_dir)
    filename = '{}_vs_{}_together'.format('_'.join(x_label.split()), '_'.join(y_label.split()))
    figpath = plots_dir + '/' + filename + '.' + format
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    idx = 0
    for model, vals in sens_dict.items():
        ax.plot(vals[0], vals[1], color=colors[idx], label='{} (Women)'.format(model.shortfilename()))
        ax.plot(nosens_dict[model][0], nosens_dict[model][1], linestyle=':', color=colors[idx], label='{} (Men)'.format(model.shortfilename()))
        idx += 1
    ax.set_xlabel(x_label + ' (' + r'$\delta$' + ')')
    ax.set_ylabel(y_label)
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax.legend(loc='lower right', bbox_to_anchor=(1, 0.5))
    ax.legend(loc='lower right')
    plt.savefig(figpath, format=format, bbox_inches='tight')
    plt.savefig(plots_dir + '/' + filename + '.pdf', format='pdf', bbox_inches='tight')

    # for i, ax_obj in enumerate([ax]):
    #     fig_legend = plt.figure(figsize=(3, 3))
    #     handles, labels = ax_obj.get_legend_handles_labels()
    #     fig_legend.legend(handles, labels, 'center', ncol=1)
    #     fig_legend.savefig(plots_dir + '/' + filename + "_legend." + format, format=format, bbox_inches='tight')
    #     fig_legend.savefig(plots_dir + '/' + filename + "_legend.pdf", format='pdf', bbox_inches='tight')

    return "{}\n\n{}".format(get_wiki_link(figpath), get_wiki_link(plots_dir + '/' + filename + "_legend." + format))

def get_regression_weights(feature_info, clf):
    '''Returns a dict which can be used to generate a table of linear/logistic regression weights'''
    assert len(feature_info) == len(clf.clf.coef_.flatten())
    heading, values, formats = ['Feature Name', 'Coefficient'], [], [None, None]
    for i in range(len(feature_info)):
        values.append([feature_info[i][0], clf.clf.coef_.flatten()[i]])
    values.append(['Intercept', clf.clf.intercept_])
    return {'col_names': heading,
            'values': values,
            'val_format': formats}

def get_feature_names(feature_info):
    '''Returns a list of feature names (str) from a list of feature_info where each 
    entry is a tuple'''
    feature_names = []
    for fname, _, _ in feature_info:
        feature_names.append(fname)
    return feature_names

def get_top_explanations(res_dir, feature_info, explanations):
    """Extracts the top k occurring explanations and returns them
    as a listsing
    """
    num_explanations = 3
    sorted_explanations = sorted(explanations.items(), key=lambda p: p[1], reverse=True)
    top_explanations = ["{} occurrences:\n{}".format(exp_count, get_conditions_str(feature_info, nec_vars, nec_vals, level=1)) for (nec_vars, nec_vals), exp_count in sorted_explanations[:num_explanations]]

    histogram_link = plot_explanation_hist(res_dir, sorted_explanations)
    return out.get_listing(top_explanations, numbered=True) + "\n" + histogram_link + "\n\n"

def get_cost_func_desc(feature_val_costs):
    """Constructs a description of the cost functions for differentfeatures"""
    feature_cost_desc = ["{}:\n{}".format(fname, get_dict_listing(
            val_costs, level=1, sort_by='val')) \
        for fname, val_costs in feature_val_costs.items()]
    cost_desc = out.get_listing(feature_cost_desc)
    return cost_desc.replace("\n\n", "\n")

def get_feature_wise_effort(feature_info, user_vector, tar_nec_vars, tar_vals, 
        cost_funcs, cost_funcs_rev, sens_group, tar_gt, target_pred, user_gt, user_pred):
    """ 
    target_gt: gound truth of role model
    target_pred: predicted label of role model
    user_gt: user's ground truth
    user_pred: user's predicted value
    """
    idx = 0
    user_indices, user_vals = cf.bin_to_index_vals(feature_info, user_vector)
    # print ("\nIndices: {},\nvals: {}\n".format(user_indices, user_vals))
    explanation = ge.get_conditions(feature_info, tar_nec_vars, tar_vals)
    # print ("\n\n{}\n\n".format(explanation))
    feature_to_effort = defaultdict(str)
    for (fname, _, _), user_ind, user_val in zip(feature_info, user_indices, user_vals):
        (new_ind, new_val) = explanation[fname] if fname in explanation else (user_ind, user_val)
        effort = max(0,max(cost_funcs[fname](user_val, new_val, sens_group), cost_funcs_rev[fname](user_val, new_val, sens_group)))
        feature_to_effort[fname] = effort
    # print (feature_to_effort)
    return feature_to_effort, explanation

def get_feature_wise_str(feature_wise_effort, explanation):
    string = "  * "
    for fname, effort in feature_wise_effort.items():
        if fname in explanation:
            string += "{}: {:.3f}; ".format(fname, effort)
    string = string[:-2]
    string += "\n"
    return string

def load_model(model, dataset_name,flipped=False):
    model_name = model.filename() if not flipped else model.filename() + '_flipped'
    if not os.path.exists('./pickled_models/{}/{}.pkl'.format(dataset_name, model_name)):
        return False, None
    else:
        return True, joblib.load('./pickled_models/{}/{}.pkl'.format(dataset_name, model_name))

def persist_model(clf, dataset_name, flipped=False):
    out.create_dir('./pickled_models')
    out.create_dir('./pickled_models/{}'.format(dataset_name))
    joblib.dump(clf, "./pickled_models/{}/{}.pkl".format(dataset_name, clf.filename() if not flipped else clf.filename() + '_flipped'))

def load_params(store_loc, key):
    with shelve.open(store_loc) as param_db:
        best_params = param_db.get(key)
        return best_params

def save_params(store_loc, key, best_params):
    with shelve.open(store_loc) as param_db:
        param_db[key] = best_params
