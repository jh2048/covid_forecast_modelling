import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from sklearn.metrics import mean_absolute_error
import lmfit
from tqdm.auto import tqdm
import pickle
import joblib
import os

from notebooks.sir_models.models import SEIRHidden
from notebooks.sir_models.fitters import HiddenCurveFitter
from notebooks.sir_models.utils import stepwise, eval_on_select_dates_and_k_days_ahead


## Data Preparation
def prepare_data(df, date_col_name, state, split_date='2020-07-29'):
    df_state = df[df.State == state]
    df_state.dropna(subset=['deaths_per_day_ma7', 'infected_per_day_ma7','recovered_per_day_ma7'], how='any', inplace=True)

    train_subset = df_state[df_state[date_col_name] <= pd.to_datetime(split_date)]
    test_subset = df_state[df_state[date_col_name]  > train_subset[date_col_name].max()]
    
    return df_state, train_subset, test_subset



def initialise_model(train_subset, population, population_size, stepwize_size = 60):
    
    weights = {
        'I': 0.25,
        'R': 0.25,
        'D': 0.5,
    }
    model = SEIRHidden(stepwise_size=stepwize_size)
    fitter = HiddenCurveFitter(
        population_size = population,
        population_density = population_density,
        new_deaths_col='deaths_per_day_ma7',
        new_cases_col='infected_per_day_ma7',
        new_recoveries_col='recovered_per_day_ma7',
        
        weights=weights,
        max_iters=1000,
        save_params_every=50,
    )
    fitter.fit(model, train_subset)
    return model, fitter





def training_model(model, train_subset):
    train_initial_conditions = model.get_initial_conditions(train_subset)
    train_t = np.arange(len(train_subset))

    (S, E, I, Iv, R, Rv, D, Dv), history = model.predict(train_t, train_initial_conditions)
    predictions = {'S':S, 'E':E, 'I':I, 'Iv':Iv, 'R':R, 'Rv':Rv, 'D':D, 'Dv':Dv}
    
    (new_exposed,
            new_infected_invisible, new_infected_visible,
            new_recovered_invisible,
            new_recovered_visible,
            new_dead_invisible, new_dead_visible) = model.compute_daily_values(S, E, I, Iv, R, Rv, D, Dv)
    
    daily_df = pd.DataFrame(
    {
        'date': train_subset.date[1:].values,
        'new_exposed': new_exposed,
        'new_infected_invisible': new_infected_invisible,
        'new_infected_visible': new_infected_visible,
        'new_recovered_invisible': new_recovered_invisible,
        'new_recovered_visible': new_recovered_visible,
        'new_dead_invisible': new_dead_invisible,
        'new_dead_visible': new_dead_visible
    },
    index=train_subset.date[1:])
    return daily_df, predictions, train_t






def forecasting(train_subset, test_subset, model, predictions_prev):
    test_t = len(train_subset) + np.arange(len(test_subset))

    (S, E, I, Iv, R, Rv, D, Dv) = predictions_prev.values()

    test_initial_conds = (S[-1], E[-1], I[-1], Iv[-1], R[-1], Rv[-1], D[-1], Dv[-1])
    (test_S, test_E, test_I, test_Iv, test_R, test_Rv, test_D, test_Dv), history = model.predict(test_t, test_initial_conds)
    predictions = {'S':test_S, 'E':test_E, 'I':test_I, 'Iv':test_Iv, 'R':test_R, 'Rv':test_Rv, 'D': test_D, 'Dv':test_Dv}
    

    (test_new_exposed,
            test_new_infected_invisible, test_new_infected_visible,
            test_new_recovered_invisible,
            test_new_recovered_visible,
            test_new_dead_invisible, test_new_dead_visible) = model.compute_daily_values(test_S, test_E, test_I, test_Iv, test_R, test_Rv, test_D, test_Dv)

    test_daily_df = pd.DataFrame(
        {
            'date': test_subset.ObservationDate[1:].values,
            'new_exposed': test_new_exposed,
            'new_infected_invisible': test_new_infected_invisible,
            'new_infected_visible': test_new_infected_visible,
            'new_recovered_invisible': test_new_recovered_invisible,
            'new_recovered_visible': test_new_recovered_visible,
            'new_dead_invisible': test_new_dead_invisible,
            'new_dead_visible': test_new_dead_visible
        },
        index=test_subset.ObservationDate[1:])

    return test_daily_df, predictions, test_t



def visualisations_training(train_subset, daily_df, D, Dv, state):
    plt.figure(figsize=(10, 7))
    plt.plot(train_subset.date, train_subset['total_deaths'], label='ground truth')
    plt.plot(train_subset.date, D, label='predicted invisible', color='black', linestyle='dashed' )
    plt.plot(train_subset.date, Dv, label='predicted visible', color='black')
    plt.legend()
    plt.title('Total deaths')
    plt.savefig(f'figures/{state}/total_deaths_prediction.png')
    plt.clf()

    plt.figure(figsize=(10, 7))
    plt.plot(daily_df.new_dead_visible, label='daily deaths_visible', color='black', linestyle='dashed')
    plt.plot(daily_df.new_dead_invisible, label='daily deaths_invisible', color='black', linestyle=':')
    plt.plot(train_subset.date, train_subset['deaths_per_day_ma7'], label='ground truth')
    plt.legend()
    plt.title('Daily deaths')
    plt.savefig(f'figures/{state}/daily_deaths_prediction.png')
    plt.clf()

    plt.figure(figsize=(10, 7))
    plt.plot(train_subset.date, train_subset['total_infected'], label='ground truth')
    plt.plot(daily_df.new_infected_visible.cumsum(), label='predicted visible', color='red')
    plt.plot(daily_df.new_infected_invisible.cumsum(), label='predicted invisible', color='red',  linestyle='dashed' )
    plt.legend()
    plt.title('Total infections')
    plt.savefig(f'figures/{state}/total_infections_prediction.png')
    plt.clf()

    print(train_subset[['date', 'infected_per_day_ma7']])
    plt.figure(figsize=(10, 7))
    plt.plot(train_subset.date, train_subset['infected_per_day_ma7'], label='ground truth')
    plt.plot(daily_df.new_infected_visible, label='daily infected_visible', color='red')
    plt.plot(daily_df.new_infected_invisible, label='daily infected_invisible', color='red', linestyle='dashed')
    plt.legend()
    plt.title('Daily infections')
    plt.savefig(f'figures/{state}/daily_infections_prediction.png')
    plt.clf()




def visulations_forecasting(train_subset, test_subset, test_daily_df, daily_df, test_D, test_Dv, test_R, test_Rv, state):
    plt.figure(figsize=(10, 7))
    plt.plot(train_subset.date, train_subset['total_deaths'], label='train ground truth')
    plt.plot(test_subset.date, test_subset['total_deaths'], label='test ground truth', color='black')
    plt.plot(test_subset.date, test_D, label='test forecasted invisible', color='black', linestyle=':')
    plt.plot(test_subset.date, test_Dv, label='test forecasted visible', color='black', linestyle='dashed')
    plt.legend()
    plt.title('Total deaths')
    plt.savefig(f'figures/{state}/daily_infections_prediction.png')
    plt.clf()

    plt.figure(figsize=(10, 7))
    plt.plot(train_subset.date, train_subset['deaths_per_day'], label='train ground truth')
    plt.plot(test_subset.date, test_subset['deaths_per_day'], label='test ground truth', color='black')
    plt.plot(test_daily_df.new_dead_invisible, label='test forecasted invisible', color='black', linestyle=':')
    plt.plot(test_daily_df.new_dead_visible, label='test forecasted visible', color='black', linestyle='dashed')
    plt.plot(daily_df.new_dead_visible, label='daily deaths_visible tain', color='black', alpha=0.5, linestyle='dashed')
    plt.legend()
    plt.title('Daily deaths')
    plt.savefig(f'figures/{state}/daily_infections_prediction.png')
    plt.clf()

    plt.figure(figsize=(10, 7))
    plt.plot(train_subset.date, train_subset['recovered_per_day'], label='train ground truth')
    plt.plot(test_subset.date, test_subset['recovered_per_day'], label='test ground truth', color='green')
    plt.plot(test_daily_df.new_recovered_invisible, label='test forecasted invisible', color='green', linestyle='dashed')
    plt.plot(test_daily_df.new_recovered_visible, label='test forecasted visible', color='green', linestyle=':')
    plt.legend()
    plt.title('Daily recoveries')
    plt.savefig(f'figures/{state}/daily_infections_prediction.png')
    plt.clf()

    plt.figure(figsize=(10, 7))
    plt.plot(train_subset.date, train_subset['total_recovered'], label='train ground truth')
    plt.plot(test_subset.date, test_subset['total_recovered'], label='test ground truth', color='green')
    plt.plot(test_subset.date, test_R, label='test forecasted invisible', color='green', linestyle=':')
    plt.plot(test_subset.date, test_Rv, label='test forecasted visible', color='green', linestyle='dashed')
    plt.legend()
    plt.title('Total recovered')
    plt.savefig(f'figures/{state}/daily_infections_prediction.png')
    plt.clf()

    plt.figure(figsize=(10, 7))
    plt.plot(train_subset.date, train_subset['infected_per_day'], label='train ground truth')
    plt.plot(test_subset.date, test_subset['infected_per_day'], label='test ground truth', color='blue')
    plt.plot(test_daily_df.new_infected_invisible, label='test forecasted invisible', color='red', linestyle=':')
    plt.plot(test_daily_df.new_infected_visible, label='test forecasted visible', color='red', linestyle='dashed')
    plt.legend()
    plt.title('Daily infections')
    plt.savefig(f'figures/{state}/daily_infections_prediction.png')
    plt.clf()



def eval_hidden_moscow(train_df, t, train_t, eval_t, population_size, population_density):
    weights = {
        'I': 0.25,
        'R': 0.25,
        'D': 0.5,
    }
    
    model = SEIRHidden()
    fitter = HiddenCurveFitter(
        population_size = population_size,
        population_density = population_density,
        new_deaths_col='deaths_per_day_ma7',
        new_cases_col='infected_per_day_ma7',
        new_recoveries_col='recovered_per_day_ma7',
        weights=weights,
        max_iters=1000,
        save_params_every=500)
    fitter.fit(model, train_df)

    train_initial_conditions = model.get_initial_conditions(train_df)
    train_states, history = model.predict(train_t, train_initial_conditions, history=False)

    test_initial_conds = [compartment[-1] for compartment in train_states]
    test_states, history = model.predict(eval_t, test_initial_conds, history=False)
        
    return model, fitter, test_states



def visualisation_validation(train_subset, daily_df, x_dates, model_pred_D, true_D, baseline_pred_D, overall_errors_model, overall_errors_baseline):

    plt.figure(figsize=(10, 7))
    plt.plot(x_dates, overall_errors_model, label='Model error')
    plt.plot(x_dates, overall_errors_baseline, label='Baseline error')
    plt.legend()
    plt.ylabel('Mean Absolute Error')
    plt.title('MAE as a function of time')


    plt.figure(figsize=(10, 7))
    plt.scatter(x_dates, [v[-1] for v in true_D], label='True dead')
    plt.scatter(x_dates, [v[-1] for v in baseline_pred_D], label='Baseline pred dead')
    plt.scatter(x_dates, [v[-1] for v in model_pred_D], label='Model pred dead')
    plt.legend()


    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot(3, 1, 1)
    plt.plot(train_subset.date, train_subset['infected_per_day'], label='Historical data on infections')
    plt.plot(daily_df.new_infected_visible, label='Model: infected visible', color='red', )
    plt.plot(daily_df.new_infected_invisible, label='Model: infected invisible', color='red', alpha=0.5, linestyle='dashed')
    plt.legend(loc="upper center")
    plt.ylabel('Infected per day')


    plt.subplot(3, 1, 2, sharex=ax1)
    plt.plot(train_subset.date, train_subset['recovered_per_day'], label='Historical data on recoveries')
    plt.plot(daily_df.new_recovered_visible, label='Model: recovered visible', color='green', )
    plt.plot(daily_df.new_recovered_invisible, label='Model: recovered invisible', color='green', alpha=0.5, linestyle='dashed')
    plt.legend(loc="upper center")
    plt.ylabel('Recovered per day')


    plt.subplot(3, 1, 3, sharex=ax1)
    plt.plot(train_subset.date, train_subset['deaths_per_day'], label='Historical data on deaths')
    plt.plot(daily_df.new_dead_visible, label='Model: deaths visible', color='black', )
    plt.plot(daily_df.new_dead_invisible, label='Model: deaths invisible', color='black', alpha=0.5, linestyle='dashed')
    plt.legend(loc="upper center")
    plt.ylabel('Deaths per day')
    plt.xlabel('Days')

    plt.tight_layout()
    plt.savefig(f'figures/{state}/daily_infected_dead_recovered_train.png')
    plt.clf()


def plot_eval_result(ix, first=False):
    eval_date = eval_dates[ix]
    train_df = train_dfs[ix]
    test_df = test_dfs[ix]
    model_preds = model_predictions[ix]

    plt.plot(train_df.date, train_df.total_deaths, label='Train data', color='black')
    plt.plot(test_df.date, test_df.total_deaths, label='Test data', color='black', linestyle='dashed')
    plt.plot(test_df.date, model_preds[7], label='Forecast', color='red')
    plt.title(f'Evaluation date: {str(eval_date.date())}')
    plt.axvline(x=eval_date, linestyle='dotted', label='Evaluation date')
    if first:
        plt.ylabel('Cumulative deaths')
    else:
        plt.setp(plt.gca().get_yticklabels(), visible=False)








if __name__ == '__main__':

    ## Load dataset
    df = pd.read_pickle('data/prepared_data.pkl')

    
    for state in df.State.unique():
        print(state)
        ## Create folder for each state
        state_filename = state.strip()
        if not os.path.exists(f'figures/{state_filename}'):
            os.mkdir(f'figures/{state_filename}')

        ## Prepare dataset
        df_state, train_subset, test_subset = prepare_data(df, 'ObservationDate', state)
        with open(f'figures/{state_filename}/log.txt', 'w') as f:
            f.write(f'Start Date: {str(min(df_state.ObservationDate))}\n')
            f.write(f'End Date: {str(max(df_state.ObservationDate))}\n')
        f.close()

        ## Initialise Model
        print(train_subset['TPopulation1July'], train_subset['TPopulation1July'].unique())
        population_size = train_subset['TPopulation1July'].unique()[0]*1000
        print('Population size:', population_size)
        population_density = train_subset['PopDensity'].unique()[0]
        model, fitter = initialise_model(train_subset, population_size, population_density)
        with open(f'figures/{state_filename}/log.txt', 'a') as f:
            print(fitter.result, file=f)
        f.close()

        ## Record incubation days
        incubation_days = model.params['incubation_days'].value
        infectious_days = model.params['infectious_days'].value
        with open(f'figures/{state_filename}/log.txt', 'a') as f:
            f.write(f'Incubation period: {incubation_days:.2f}\n')
            f.write(f'Infectious period: {infectious_days:.2f}')
        f.close()

        ## train_model
        daily_df, predictions, train_t = training_model(model=model, train_subset=train_subset)
        visualisations_training(train_subset, daily_df, predictions['D'], predictions['Dv'], state_filename)

        ## Forecasting
        test_daily_df, new_predictions, test_t = forecasting(train_subset, test_subset, model, predictions)
        visulations_forecasting(train_subset, test_subset, test_daily_df, daily_df, new_predictions['D'], new_predictions['Dv'], new_predictions['R'], new_predictions['Rv'], state_filename)

        ## 30 Day Eval
        K = 30
        last_day = df_state.date.max() - pd.to_timedelta(K, unit='D')
        eval_dates = pd.date_range(start='2020-03-01', end=last_day)[::20]

        models, fitters, model_predictions, train_dfs, test_dfs = eval_on_select_dates_and_k_days_ahead(df_state,
                                                                                            eval_func=eval_hidden_moscow, 
                                                                                            eval_dates=eval_dates, 
                                                                                            k=K,
                                                                                            population_size=population_size,
                                                                                            population_density=population_density)
        x_dates = [tdf.ObservationDate.iloc[-1] for tdf in test_dfs]
        model_pred_D = [pred[7] for pred in model_predictions]
        true_D = [tdf.total_deaths.values for tdf in test_dfs]
        baseline_pred_D = [[tdf.iloc[-1].total_deaths]*K for tdf in train_dfs]
        overall_errors_model = [mean_absolute_error(true, pred) for true, pred in zip(true_D, model_pred_D)]
        overall_errors_baseline = [mean_absolute_error(true, pred) for true, pred in zip(true_D, baseline_pred_D)]

        with open(f'figures/{state_filename}/model_error_rate.txt', 'w') as f:
            f.write(f'\nMean overall error baseline: {np.mean(overall_errors_baseline).round(3)}')
            f.write(f'\nMean overall error model: {np.mean(overall_errors_model).round(3)}')
        f.close()
        visualisation_validation(train_subset, daily_df, x_dates, model_pred_D, true_D, baseline_pred_D, overall_errors_model, overall_errors_baseline)

        try:
            ## Evaluate dates
            fig = plt.figure(figsize=(20, 7))

            ax1 = plt.subplot(1, 3, 1)
            ix = 3
            plot_eval_result(ix, first=True)

            plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
            ix = 10
            plot_eval_result(ix)

            ax3 = plt.subplot(1, 3, 3, sharex=ax1, sharey=ax1)
            ix = 13
            plot_eval_result(ix)
        except:
            pass

