import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import experiment as exp

def compute_metrics(Y, classification_res, prediction_task, print_stats=False):
    """Compute performance metrics of a for predicted labels classification_res and ground truth labels Y"""

    if prediction_task == exp.CLASSIFICATION:
        num_rows = len(Y)
        num_pos_data = np.count_nonzero(Y)
        num_pos_clas = np.count_nonzero(classification_res)
        num_correct_clas = np.count_nonzero(np.equal(classification_res, Y)) ### True positive + True Negative
        num_true_pos_clas = np.count_nonzero(np.logical_and(classification_res==1, Y==1)) ## True positive
        num_false_pos_clas = np.count_nonzero(np.logical_and(classification_res==1, Y ==0))
        num_false_neg_clas = np.count_nonzero(np.logical_and(classification_res==0, Y==1))
        num_true_neg_clas = np.count_nonzero(np.logical_and(classification_res==0, Y ==0))

        #if print_stats:
        #    print('num rows: ', num_rows, ', num_pos_data: ', num_pos_data, ', num_pos_clas: ', num_pos_clas,
        #          ', num_correct_clas: ', num_correct_clas, ', num_true_pos_clas: ', num_true_pos_clas, 'num_false_pos_clas',num_false_pos_clas, 'num_false_neg_clas',num_false_neg_clas)

        accuracy = num_correct_clas / num_rows
        recall = (num_true_pos_clas / num_pos_data) if num_pos_data > 0 else 1.0
        precision = (num_true_pos_clas / num_pos_clas) if num_pos_clas > 0 else 1.0
        fpr = num_false_pos_clas/(num_false_pos_clas+ num_true_neg_clas) if (num_false_pos_clas + num_true_neg_clas) >0 else 1.0
        fnr = num_false_neg_clas /(num_false_neg_clas + num_true_pos_clas) if (num_false_neg_clas + num_true_pos_clas) > 0 else 1.0

        if print_stats:
            print("------------")
            print("Accuracy:", accuracy)
            print("Recall:", recall)
            print("Precision:", precision)
            print("------------")
        return {'Accuracy': accuracy, 'Recall': recall, 'Prescision': precision, 'FPR': fpr, 'FNR': fnr}
    elif prediction_task == exp.REGRESSION:
        mae = mean_absolute_error(Y, classification_res)
        mse = mean_squared_error(Y, classification_res)
        if print_stats:
            print("------------")
            print("Mean Avg Error:", mae)
            print("Mean Squared Error:", mse)
            print("------------")
        return {'Mean Avg Error': mae, 'Mean Squared Error': mse}


def eval_model(model, X, Y, prediction_task, print_stats=True):
    return compute_metrics(Y, model.predict(X), prediction_task, print_stats)

def get_disparity_measures(y_true, y_pred, sens_group, efforts_sens, efforts_nosens, prediction_task, return_heading_and_formats=False):
    if prediction_task == exp.REGRESSION:
        heading = ["MSE <<BR>> disp = MSE(Y_pred_s, Y_GT_s|)/|S| - <<BR>> MSE(Y_pred_~s, Y_GT_~s|)/|~S|",
                    "MAE <<BR>> disp = MAE(Y_pred_s, Y_GT_s|)/|S| - <<BR>> MAE(Y_pred_~s, Y_GT_~s|)/|~S|",
                    "Positive Residual <<BR>> Difference",
                    "Negative Residual <<BR>> Difference",
                    "Effort-Reward <<BR>> (avg. utility for sens, avg utility for non_sens) <<BR>> disp = abs( avg. utility for sens - avg utility for non_sens)"]
        formats = [None, None, None, None, None]
        if efforts_sens is None or efforts_nosens is None:
            heading = heading[:-1]
            formats = formats[:-1]
            values = [disparity_mse(y_true, y_pred, sens_group), disparity_mae(y_true, y_pred, sens_group),
                    positive_residual(y_true, y_pred, sens_group), negative_residual(y_true, y_pred, sens_group)]
        else:
            values = [disparity_mse(y_true, y_pred, sens_group), disparity_mae(y_true, y_pred, sens_group),
                    positive_residual(y_true, y_pred, sens_group), negative_residual(y_true, y_pred, sens_group),
                    disparity_utility(efforts_sens, efforts_nosens)]
    elif prediction_task == exp.CLASSIFICATION:
        heading = ["Accuracy <<BR>> (|Y_pred_s and Y_GT_s|/|S|, |Y_predicted_~s and Y_GT_~s|/|~S|) <<BR>> disp = abs( (|Y_pred_s and Y_GT_s|/|S|) - (|Y_predicted_~s and Y_GT_~s|/|~S|) )",
                "Impact <<BR>> (|Y_predicted_s|/|S|, |Y_predicted_~s|/|~S|) <<BR>> disp = abs( (|Y_predicted_s|/|S|) - (|Y_predicted_~s|/|~S|) )",
                "FPR <<BR>> ( P(Y_pred = 1 | S = 1, Y_GT = 0), P(Y_pred = 1 | S = 0, Y_GT = 0) ) <<BR>> disp = abs( P(Y_pred = 1 | S = 1, Y_GT = 0) - P(Y_pred = 1 | S = 0, Y_GT = 0) )",
                "TPR <<BR>> ( P(Y_pred = 1 | S = 1, Y_GT = 1), P(Y_pred = 1 | S = 0, Y_GT = 1) ) <<BR>> disp = abs( P(Y_pred = 1 | S = 1, Y_GT = 1) - P(Y_pred = 1 | S = 0, Y_GT = 1) )",
                "FOR <<BR>> ( P(Y_GT = 1 | S = 1, Y_pred = 0), P(Y_GT = 1 | S = 0, Y_pred = 0) ) <<BR>> disp = abs( P(Y_GT = 1 | S = 1, Y_pred = 0) - P(Y_GT = 1 | S = 0, Y_pred = 0) )",
                "Precision <<BR>> ( P(Y_GT = 1 | S = 1, Y_pred = 1), P(Y_GT = 1 | S = 0, Y_pred = 1) ) <<BR>> disp = abs( P(Y_GT = 1 | S = 1, Y_pred = 1)/ P(Y_GT = 1 | S = 0, Y_pred = 1) )",
                "Effort-Reward <<BR>> (avg. utility for sens, avg utility for non_sens) <<BR>> disp = abs( avg. utility for sens - avg utility for non_sens)"
                ]
        formats = [None, None, None, None, None, None, None]
        if efforts_sens is None or efforts_nosens is None:
            heading = heading[:-1]
            formats = formats[:-1]
            values = [disparity_accuracy(y_true, y_pred, sens_group), disparity_impact(y_pred, sens_group), disparity_fpr(y_true, y_pred, sens_group), 
                    disparity_tpr(y_true, y_pred, sens_group), disparity_for(y_true, y_pred, sens_group), disparity_precision(y_true, y_pred, sens_group)]
        else:
            values = [disparity_accuracy(y_true, y_pred, sens_group), disparity_impact(y_pred, sens_group), disparity_fpr(y_true, y_pred, sens_group), 
                    disparity_tpr(y_true, y_pred, sens_group), disparity_for(y_true, y_pred, sens_group), disparity_precision(y_true, y_pred, sens_group),
                    disparity_utility(efforts_sens, efforts_nosens)]
    if return_heading_and_formats:
        return heading, formats, values
    else:
        return values

def positive_residual(y_true, y_pred, sens_group):
    y_true_s, y_pred_s = y_true[sens_group], y_pred[sens_group]
    y_true_ns, y_pred_ns = y_true[~sens_group], y_pred[~sens_group]
    g1_pos = np.where(y_pred_s > y_true_s)[0] # take only points with y_pred > y_true
    g2_pos = np.where(y_pred_ns > y_true_ns)[0]
    g1_sum = (y_pred_s - y_true_s)[g1_pos] # sum of this will be numerator
    g1_sum[np.where(g1_sum < 0)[0]] = 0 # max (0, y_pred - y_true)
    g2_sum = (y_pred_ns - y_true_ns)[g2_pos]
    g2_sum[np.where(g2_sum < 0)[0]] = 0
    return abs(np.sum(g1_sum)/len(g1_pos) - np.sum(g2_sum)/len(g2_pos))

def negative_residual(y_true, y_pred, sens_group):
    y_true_s, y_pred_s = y_true[sens_group], y_pred[sens_group]
    y_true_ns, y_pred_ns = y_true[~sens_group], y_pred[~sens_group]
    g1_neg = np.where(y_pred_s < y_true_s)[0] # take only points with y_pred < y_true
    g2_neg = np.where(y_pred_ns < y_true_ns)[0]
    g1_sum = (y_true_s - y_pred_s)[g1_neg] # sum of this will be numerator
    g1_sum[np.where(g1_sum < 0)[0]] = 0 # max (0, y_true - y_pred)
    g2_sum = (y_true_ns - y_pred_ns)[g2_neg]
    g2_sum[np.where(g2_sum < 0)[0]] = 0
    return abs(np.sum(g1_sum)/len(g1_neg) - np.sum(g2_sum)/len(g2_neg))

def disparity_accuracy(y_true, y_pred, sens_group):
    correct_and_sens = np.count_nonzero(np.logical_and(y_true == y_pred, sens_group))
    correct_and_nosens = np.count_nonzero(np.logical_and(y_true == y_pred, ~sens_group))
    sens = correct_and_sens/np.count_nonzero(sens_group)
    nosens = correct_and_nosens/np.count_nonzero(~sens_group)
    # ratio = sens/nosens
    return "%.3f (%.3f, %.3f)"%(abs(sens - nosens), sens, nosens)

def disparity_impact(y_pred, sens_group):
    sens = np.count_nonzero(np.logical_and(y_pred, sens_group))/np.count_nonzero(sens_group)
    nosens = np.count_nonzero(np.logical_and(y_pred, ~sens_group))/np.count_nonzero(~sens_group)
    # ratio = sens/nosens
    return "%.3f (%.3f, %.3f)"%(abs(sens - nosens), sens, nosens)

def disparity_fpr(y_true, y_pred, sens_group):
    y_true_s = np.logical_and(~y_true, sens_group)
    y_true_ns = np.logical_and(~y_true, ~sens_group)
    fp_s = np.count_nonzero(np.logical_and(y_pred, y_true_s))/np.count_nonzero(y_true_s) if np.count_nonzero(y_true_s) != 0 else float('inf')
    fp_ns = np.count_nonzero(np.logical_and(y_pred, y_true_ns))/np.count_nonzero(y_true_ns) if np.count_nonzero(y_true_ns) != 0 else float('inf')
    # ratio = fp_s/fp_ns
    return "%.3f (%.3f, %.3f)"%(abs(fp_s - fp_ns), fp_s, fp_ns)

def disparity_tpr(y_true, y_pred, sens_group):
    y_true_s = np.logical_and(y_true, sens_group)
    y_true_ns = np.logical_and(y_true, ~sens_group)
    tp_s = np.count_nonzero(np.logical_and(y_pred, y_true_s))/np.count_nonzero(y_true_s) if np.count_nonzero(y_true_s) != 0 else float('inf')
    tp_ns = np.count_nonzero(np.logical_and(y_pred, y_true_ns))/np.count_nonzero(y_true_ns) if np.count_nonzero(y_true_ns) != 0 else float('inf')
    # ratio = tp_s/tp_ns
    return "%.3f (%.3f, %.3f)"%(abs(tp_s - tp_ns), tp_s, tp_ns)

def disparity_for(y_true, y_pred, sens_group):
    y_pred_s = np.logical_and(~y_pred, sens_group)
    y_pred_ns = np.logical_and(~y_pred, ~sens_group)
    tp_s = np.count_nonzero(np.logical_and(y_true, y_pred_s))/np.count_nonzero(y_pred_s) if np.count_nonzero(y_pred_s) != 0 else float('inf')
    tp_ns = np.count_nonzero(np.logical_and(y_true, y_pred_ns))/np.count_nonzero(y_pred_ns) if np.count_nonzero(y_pred_ns) != 0 else float('inf')
    # ratio = tp_s/tp_ns
    return "%.3f (%.3f, %.3f)"%(abs(tp_s - tp_ns), tp_s, tp_ns)

def disparity_precision(y_true, y_pred, sens_group):
    y_pred_s = np.logical_and(y_pred, sens_group)
    y_pred_ns = np.logical_and(y_pred, ~sens_group)
    tp_s = np.count_nonzero(np.logical_and(y_true, y_pred_s))/np.count_nonzero(y_pred_s) if np.count_nonzero(y_pred_s) != 0 else float('inf')
    tp_ns = np.count_nonzero(np.logical_and(y_true, y_pred_ns))/np.count_nonzero(y_pred_ns) if np.count_nonzero(y_pred_ns) != 0 else float('inf')
    # ratio = tp_s/tp_ns
    return "%.3f (%.3f, %.3f)"%(abs(tp_s - tp_ns), tp_s, tp_ns)

def disparity_mae(y_true, y_pred, sens_group):
    y_pred_s, y_true_s = y_pred[sens_group], y_true[sens_group]
    y_pred_ns, y_true_ns = y_pred[~sens_group], y_true[~sens_group]
    mae_s = mean_absolute_error(y_true_s, y_pred_s)
    mae_ns = mean_absolute_error(y_true_ns, y_pred_ns)
    return "%.3f (%.3f, %.3f)"%(abs(mae_s - mae_ns), mae_s, mae_ns)

def disparity_mse(y_true, y_pred, sens_group):
    y_pred_s, y_true_s = y_pred[sens_group], y_true[sens_group]
    y_pred_ns, y_true_ns = y_pred[~sens_group], y_true[~sens_group]
    mse_s = mean_squared_error(y_true_s, y_pred_s)
    mse_ns = mean_squared_error(y_true_ns, y_pred_ns)
    return "%.3f (%.3f, %.3f)"%(abs(mse_s - mse_ns), mse_s, mse_ns)    

def disparity_utility(utility_sens, utility_nosens):
    # ratio = utility_sens/utility_nosens
    return "%.3f (%.3f, %.3f)"%(abs(utility_sens - utility_nosens), utility_sens, utility_nosens)