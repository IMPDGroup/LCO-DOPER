import datetime
import pandas as pd
import numpy as np
import streamlit as st
import torch
import pyro
import pickle
import plotly.express as px


torch.classes.__path__ = []


class BNN(pyro.nn.PyroModule):
    def __init__(self, in_features, out_features, n_layers, prior_scale_weight, prior_scale_bias, nodes_list):
        super().__init__()
        
        self.n_layers = n_layers
        self.prior_scale_weight = prior_scale_weight
        self.prior_scale_bias = prior_scale_bias
        self.layers = torch.nn.ModuleList()
        self.activation = torch.nn.ReLU()

        # Get the suggested number of nodes for each layer
        self.nodes = [in_features]
        for i in range(self.n_layers):
            nodes = nodes_list[i]
            self.nodes.append(nodes)
        self.nodes.append(out_features)
        
        # Define Bayesian Linear layers
        layer_list = [pyro.nn.PyroModule[torch.nn.Linear](self.nodes[i-1], self.nodes[i]) for i in range(1, len(self.nodes))]
        self.layers = pyro.nn.PyroModule[torch.nn.ModuleList](layer_list)
        for i, layer in enumerate(self.layers):
            layer.weight = pyro.nn.PyroSample(pyro.distributions.Normal(0., self.prior_scale_weight).expand([self.nodes[i+1], self.nodes[i]]).to_event(2))
            layer.bias = pyro.nn.PyroSample(pyro.distributions.Normal(0., self.prior_scale_bias).expand([self.nodes[i+1]]).to_event(1))

    def forward(self, x, y=None):
        # Reshape the input
        x = x.reshape(-1, self.nodes[0])
        
        # Pass through hidden layers
        x = self.activation(self.layers[0](x))
        for layer in self.layers[1:-1]:
            x = self.activation(layer(x))
        mu = self.layers[-1](x).squeeze()

        # Bayesian inference for output uncertainty
        sigma = pyro.sample('sigma', pyro.distributions.Gamma(0.5, 1.))

        # Observational likelihood
        with pyro.plate('data', x.shape[0]):
            obs = pyro.sample('obs', pyro.distributions.Normal(mu, sigma * sigma).to_event(1), obs=y)

        return mu


def BNN_predictor(data_pre):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the x and y scaler
    with open('res/scaler_x_BNN.pkl', 'rb') as f:
        x_scaler = pickle.load(f)
    with open('res/scaler_y_BNN.pkl', 'rb') as f:
        y_scaler = pickle.load(f)

    # Load the train data
    with open('res/train_x_BNN.pkl', 'rb') as f:
        train_x = pickle.load(f)
    with open('res/train_y_BNN.pkl', 'rb') as f:
        train_y = pickle.load(f)
    train_x = torch.FloatTensor(train_x).to(DEVICE)
    train_y = torch.FloatTensor(train_y).to(DEVICE)

    # Get the input data to predict
    x_pre = data_pre[['atomic_number_A','atomic_number_B','ionic_radius_A','ionic_radius_B','dopant_concentration_A','dopant_concentration_B']].to_numpy()
    x_pre = x_scaler.transform(x_pre)
    x_pre = torch.FloatTensor(x_pre).to(DEVICE)

    # Define the model
    model_path = 'res/model_BNN.pkl'
    IN_FEATURES = 6
    OUT_FEATURES = 3
    N_LAYERS = 2
    PRIOR_SCALE_WEIGHT = 1.1127440261121575
    PRIOR_SCALE_BIAS = 0.21609393343755173
    NODES_LIST = [26, 44]
    LR = 0.009945018994994233
    PREIDCTIVE_SAMPLES = 1000
    torch.manual_seed(0)
    model = BNN(in_features=IN_FEATURES, out_features=OUT_FEATURES, n_layers=N_LAYERS, prior_scale_weight=PRIOR_SCALE_WEIGHT, prior_scale_bias=PRIOR_SCALE_BIAS, nodes_list=NODES_LIST).to(DEVICE)
    # Define variational guide
    guide = pyro.infer.autoguide.AutoDiagonalNormal(model)
    # Optimizer
    optimizer = pyro.optim.Adam({"lr": LR})
    # Loss function
    loss_func = pyro.infer.Trace_ELBO()
    # Stochastic Variational Inference
    svi = pyro.infer.SVI(model, guide, optimizer, loss=loss_func)
    # Predictive
    predictive = pyro.infer.Predictive(model, guide=guide, num_samples=PREIDCTIVE_SAMPLES)
    # Load the model
    pyro.clear_param_store()
    from functools import partial
    torch.load = partial(torch.load, weights_only=False)
    pyro.get_param_store().load(model_path)
    pyro.module('model', model, update_module_params=True)
    # Activate the model
    preds_act = predictive(train_x, train_y)

    # Get the predicted data
    with torch.no_grad():
        preds = predictive(x_pre)
        y_pre = preds['obs'].mean(0).cpu().numpy()
        y_std = preds['obs'].std(0).cpu().numpy()
        y_pre = y_scaler.inverse_transform(y_pre)
        y_std = y_std * y_scaler.scale_
        data_pred_BNN = pd.concat([data_pre, pd.DataFrame(y_pre, columns=['F', 'lattice_distortion', 'atomic_distortion'])], axis=1)
        data_pred_BNN = data_pred_BNN[['atomic_number_A', 'atomic_number_B', 'ionic_radius_A', 'ionic_radius_B', 'dopant_concentration_A', 'dopant_concentration_B', 'oxygen_vacancy_concentration', 'T', 'F', 'lattice_distortion', 'atomic_distortion']]

    return data_pred_BNN


def get_dopant_data(dopant_A, dopant_B, dopant_A_conc, dopant_B_conc, oxygen_vacancy_concentration, T):
    # Define the atomic numbers and ionic radii for the dopants
    atomic_numbers = {'Mg': 12, 'Ca': 20, 'Sr': 38, 'Ba': 56, 'Ce': 58, 
                    'Pr': 59, 'Nd': 60, 'Sm': 62, 'Gd': 64, 'Sc': 21, 
                    'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 
                    'Ni': 28, 'Cu': 29, 'Zn': 30, 'Al': 13, 'Ga': 31}
    ionic_radii = {'Mg': 0.890, 'Ca': 1.340, 'Sr': 1.440, 'Ba': 1.610, 'Ce': 1.340, 
                'Pr': 1.179, 'Nd': 1.270, 'Sm': 1.240, 'Gd': 1.107, 'Sc': 0.745, 
                'Ti': 0.670, 'V': 0.640, 'Cr': 0.615, 'Mn': 0.645, 'Fe': 0.645, 
                'Ni': 0.600, 'Cu': 0.540, 'Zn': 0.740, 'Al': 0.535, 'Ga': 0.620}
    
    # Get the atomic number and ionic radius for A-site dopant
    if dopant_A != 'None':
        atomic_number_A = atomic_numbers[dopant_A]
        ionic_radius_A = ionic_radii[dopant_A]
        dopant_concentrations_A = np.arange(dopant_A_conc[0], dopant_A_conc[1] + 1, 1)
    else:
        atomic_number_A = 0
        ionic_radius_A = 0
        dopant_concentrations_A = [0]
    
    # Get the atomic number and ionic radius for B-site dopant
    if dopant_B != 'None':
        atomic_number_B = atomic_numbers[dopant_B]
        ionic_radius_B = ionic_radii[dopant_B]
        dopant_concentrations_B = np.arange(dopant_B_conc[0], dopant_B_conc[1] + 1, 1)
    else:
        atomic_number_B = 0
        ionic_radius_B = 0
        dopant_concentrations_B = [0]

    # Create a DataFrame to store the data
    data_pre = pd.DataFrame()
    for i, A_conc in enumerate(dopant_concentrations_A):
        for j, B_conc in enumerate(dopant_concentrations_B):
            data_row = pd.DataFrame({
                'atomic_number_A': [atomic_number_A],
                'atomic_number_B': [atomic_number_B],
                'ionic_radius_A': [ionic_radius_A],
                'ionic_radius_B': [ionic_radius_B],
                'dopant_concentration_A': [A_conc/100],
                'dopant_concentration_B': [B_conc/100],
                'oxygen_vacancy_concentration': [oxygen_vacancy_concentration/100],
                'T': [T],
            })
            data_pre = pd.concat([data_pre, data_row], ignore_index=True)

    return data_pre


def ANN_predictor(data_pred_BNN):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the scaler
    with open('res/scaler_x_ANN.pkl', 'rb') as f:
        x_scaler = pickle.load(f)
    with open('res/scaler_y_ANN.pkl', 'rb') as f:
        y_scaler = pickle.load(f)
    
    # Load the model
    model = torch.load('res/model_ANN.pkl', weights_only=False)
    
    # Get the input data to predict
    x_pre = data_pred_BNN[['oxygen_vacancy_concentration', 'T', 'F', 'lattice_distortion', 'atomic_distortion']].to_numpy()
    x_pre = x_scaler.transform(x_pre)
    x_pre = torch.FloatTensor(x_pre).to(DEVICE)
   
    # Predict the data
    with torch.no_grad():
        y_pre = model(x_pre)
        y_pre = y_scaler.inverse_transform(y_pre)
    data_pred_ANN = data_pred_BNN.copy()
    data_pred_ANN['D'] = y_pre

    return data_pred_ANN


def data_display(data):
    data_dis = pd.DataFrame()
    data_dis['Dopant A'] = data['atomic_number_A'].map({12: 'Mg', 20: 'Ca', 38: 'Sr', 56: 'Ba', 58: 'Ce', 59: 'Pr', 60: 'Nd', 62: 'Sm', 64: 'Gd', 0: 'None'})
    data_dis['Dopant B'] = data['atomic_number_B'].map({21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 28: 'Ni', 29: 'Cu', 30: 'Zn', 13: 'Al', 31: 'Ga', 0: 'None'})
    data_dis['Dopant Concentration A (at.%)'] = data['dopant_concentration_A'] * 100
    data_dis['Dopant Concentration B (at.%)'] = data['dopant_concentration_B'] * 100
    data_dis['Oxygen Vacancy Concentration (at.%)'] = data['oxygen_vacancy_concentration'] * 100
    data_dis['Temperature (K)'] = data['T']
    data_dis['F (eV/atom)'] = data['F']
    data_dis['Lattice Distortion (%)'] = data['lattice_distortion'] * 100
    data_dis['Atomic Distortion (Å)'] = data['atomic_distortion']
    data_dis['D (cm^2/s)'] = data['D']
    # Set the number format
    data_dis['Dopant Concentration A (at.%)'] = data_dis['Dopant Concentration A (at.%)'].apply(lambda x: f'{x:.2f}')
    data_dis['Dopant Concentration B (at.%)'] = data_dis['Dopant Concentration B (at.%)'].apply(lambda x: f'{x:.2f}')
    data_dis['Oxygen Vacancy Concentration (at.%)'] = data_dis['Oxygen Vacancy Concentration (at.%)'].apply(lambda x: f'{x:.2f}')
    data_dis['Temperature (K)'] = data_dis['Temperature (K)'].apply(lambda x: f'{x:d}')
    data_dis['F (eV/atom)'] = data_dis['F (eV/atom)'].apply(lambda x: f'{x:.4f}')
    data_dis['Lattice Distortion (%)'] = data_dis['Lattice Distortion (%)'].apply(lambda x: f'{x:.2f}')
    data_dis['Atomic Distortion (Å)'] = data_dis['Atomic Distortion (Å)'].apply(lambda x: f'{x:.4f}')
    data_dis['D (cm^2/s)'] = data_dis['D (cm^2/s)'].apply(lambda x: f'{x:.4e}')
    # Reorder the columns
    data_dis = data_dis[['Dopant A', 'Dopant Concentration A (at.%)', 'Dopant B', 'Dopant Concentration B (at.%)', 'Oxygen Vacancy Concentration (at.%)', 'Temperature (K)', 'F (eV/atom)', 'Lattice Distortion (%)', 'Atomic Distortion (Å)', 'D (cm^2/s)']]
    return data_dis


def main():
    # Set the page config
    st.set_page_config(
        page_title='LCO-DOPER', 
        layout='wide', 
        page_icon=':material/apps:', 
        menu_items={
        'Get Help': 'https://github.com/aguang5241/LCO-DOPER',
        'Report a bug': "mailto:gliu4@wpi.edu",
        'About': "# LCO-DOPER"
        "\n**AI Powered Materials Innovation**  "
        "\n\n*Created by Guangchen Liu (gliu4@wpi.edu)*  "
        "\n*IMPD Group, Worcester Polytechnic Institute, MA USA*",
    })
    
    # Add app logo to the sidebar
    st.sidebar.image('res/logo.png', use_container_width=True)
    # Set the sidebar title
    st.sidebar.title('LCO-DOPER  [![GitHub stars](https://img.shields.io/github/stars/aguang5241/LCO-DOPER?style=social)](https://github.com/aguang5241/LCO-DOPER)')
    # Add a description to the sidebar
    st.sidebar.markdown('An application for analyzing dopant effects on LaCoO<sub>3</sub> (LCO), supporting composition optimization and material performance enhancement.', unsafe_allow_html=True)
    # Add contact information: gliu4@wpi.edu
    st.sidebar.divider()
    st.sidebar.markdown('For any questions or suggestions, please contact:')
    st.sidebar.markdown('[![Email](https://img.shields.io/badge/Email-yzhong@wpi.edu-white?logo=mail.ru&logoColor=white)](mailto:yzhong@wpi.edu) [![Email](https://img.shields.io/badge/Email-gliu4@wpi.edu-white?logo=mail.ru&logoColor=white)](mailto:gliu4@wpi.edu)')
    
    # Add a title to the main page
    st.title('LCO-DOPER  [![GitHub stars](https://img.shields.io/github/stars/aguang5241/LCO-DOPER?style=social)](https://github.com/aguang5241/LCO-DOPER)')
    # Add a description to the main page
    st.markdown('An application for analyzing dopant effects on LaCoO<sub>3</sub> (LCO), supporting composition optimization and material performance enhancement.', unsafe_allow_html=True)
    # Add a subheader for the dopant selection
    st.divider()
    st.subheader('Dopants Selection:')
    # Set two columns for the main content
    col1, col2 = st.columns(2, border=False, gap='medium')

    # Set the 1st column for A-site Dopant
    with col1:
        # Add a container for all the A-site dopant options (using pills)
        with st.container(border=True):
            dopant_A = st.pills(
                label='Please select a dopant for the A-site (La):',
                selection_mode='single',
                options=['Mg', 'Ca', 'Sr', 'Ba', 'Ce', 'Pr', 'Nd', 'Sm', 'Gd', 'None'],
                default='Mg',
            )
            # Add a range slider for the A-site dopant concentration
            st.divider()
            if dopant_A != 'None':
                dopant_A_conc = st.slider(
                    label=f'Please select the concentration of {dopant_A} (at.%):',
                    min_value=1,
                    max_value=100,
                    value=(1, 10),
                    step=1,
                )
            else:
                dopant_A_conc = (0, 0)
            # Show the selected A-site dopant and its concentration
            if dopant_A != 'None':
                st.divider()
                st.write(f'A-site Dopant: {dopant_A} with concentration range: {dopant_A_conc[0]} - {dopant_A_conc[1]} at.%')
            else:
                st.write('No A-site dopant selected!')
            
    # Set the 2nd column for B-site Dopant
    with col2:
        # Add a container for all the A-site dopant options (using pills)
        with st.container(border=True):
            dopant_B = st.pills(
                label='Please select a dopant for the B-site (Co):',
                selection_mode='single',
                options=['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'Al', 'Ga', 'None'],
                default='Sc',
            )
            # Add a range slider for the A-site dopant concentration
            st.divider()
            if dopant_B != 'None':
                dopant_B_conc = st.slider(
                    label=f'Please select the concentration of {dopant_B} (at.%):',
                    min_value=1,
                    max_value=100,
                    value=(1, 10),
                    step=1,
                )
            else:
                dopant_B_conc = (0, 0)
            # Show the selected B-site dopant and its concentration
            if dopant_B != 'None':
                st.divider()
                st.write(f'B-site Dopant: {dopant_B} with concentration range: {dopant_B_conc[0]} - {dopant_B_conc[1]} at.%')
            else:
                st.write('No B-site dopant selected!')

    # Add a container for system conditions
    st.subheader('System Conditions:')
    # Add two columns for the temperature and oxygen vacancy concentration
    col3, col4 = st.columns(2, border=True, gap='medium')
    
    # Set the 1st column for oxygen vacancy concentration
    with col3:
        oxygen_vacancy_concentration = st.slider(
            label='Please select the oxygen vacancy concentration (at.%):',
            min_value=0.0,
            max_value=5.0,
            value=1.0,
            step=0.01,
        )
        # Show the selected oxygen vacancy concentration
        st.divider()
        st.write(f'Oxygen Vacancy Concentration: {oxygen_vacancy_concentration} at.%')
    
    # Set the 2nd column for temperature
    with col4:
        T = st.slider(
            label='Please select the temperature (K):',
            min_value=1000,
            max_value=2500,
            value=1500,
            step=10,
        )
        # Show the selected temperature
        st.divider()
        st.write(f'Temperature: {T} K')

    # Add a button to submit the selections
    st.divider()
    if st.button('Predict', type='primary', use_container_width=True):
        # Get the dopant data
        data_pre_BNN = get_dopant_data(dopant_A, dopant_B, dopant_A_conc, dopant_B_conc, oxygen_vacancy_concentration, T)
        # Predict the data using BNN
        data_pred_BNN = BNN_predictor(data_pre_BNN)
        # Predict the data using ANN
        data_pred_ANN = ANN_predictor(data_pred_BNN)
        # Process the predicted data for display
        st.session_state['data_pred_dis'] = data_display(data_pred_ANN)
    
    # Show the predicted data
    if 'data_pred_dis' in st.session_state:
        data_dis = st.session_state['data_pred_dis']
        # Show the predicted data
        if st.checkbox('Show predicted data', value=False):
            st.subheader('Predicted Data:')
            st.write(st.session_state['data_pred_dis'])
            # Create a time stamp for the file name
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

            st.download_button(
                label='Download predicted data',
                data=st.session_state['data_pred_dis'].to_csv(index=False),
                file_name='LCO-DOPER_' + timestamp + '.csv',
                mime='text/csv',
                icon=":material/download:"
            )
            st.divider()

        # Make the dopant concentration columns numeric
        data_dis['Dopant Concentration A (at.%)'] = pd.to_numeric(data_dis['Dopant Concentration A (at.%)'])
        data_dis['Dopant Concentration B (at.%)'] = pd.to_numeric(data_dis['Dopant Concentration B (at.%)'])
        data_dis['Oxygen Vacancy Concentration (at.%)'] = pd.to_numeric(data_dis['Oxygen Vacancy Concentration (at.%)'])
        data_dis['Temperature (K)'] = pd.to_numeric(data_dis['Temperature (K)'])
        data_dis['F (eV/atom)'] = pd.to_numeric(data_dis['F (eV/atom)'])
        data_dis['Lattice Distortion (%)'] = pd.to_numeric(data_dis['Lattice Distortion (%)'])
        data_dis['Atomic Distortion (Å)'] = pd.to_numeric(data_dis['Atomic Distortion (Å)'])
        data_dis['D (cm^2/s)'] = pd.to_numeric(data_dis['D (cm^2/s)'])
        
        # Set 2 columns for the heatmaps of F (eV/atom) and D (cm^2/s)
        st.subheader('Predicted Data Visualization:')
        col5, col6 = st.columns(2, border=True, gap='medium')
        
        # Set the 1st column for F (eV/atom)
        with col5:
            st.markdown('**Forming Energy (eV/atom):**', )
            # Create a heatmap for F (eV/atom)
            heatmap_F = data_dis.pivot_table(index='Dopant Concentration A (at.%)', columns='Dopant Concentration B (at.%)', values='F (eV/atom)')
            # Create the Plotly heatmap
            fig = px.imshow(
                heatmap_F,
                color_continuous_scale="Viridis",
                labels=dict(x=f'{dopant_B} (at.%)', y=f'{dopant_A} (at.%)'),
            )
            fig.update_coloraxes(
                colorbar_tickformat=".2f"
            )
            # Display in Streamlit
            st.plotly_chart(fig, use_container_width=True)
        
        # Set the 2nd column for D (cm^2/s)
        with col6:
            st.markdown('**Diffusion Coefficient (cm$^2$/s):**')
            # Create a heatmap for D (cm^2/s)
            heatmap_D = data_dis.pivot_table(index='Dopant Concentration A (at.%)', columns='Dopant Concentration B (at.%)', values='D (cm^2/s)')
            # Create the Plotly heatmap
            fig = px.imshow(
                heatmap_D,
                color_continuous_scale="Viridis",
                labels=dict(x=f'{dopant_B} (at.%)', y=f'{dopant_A} (at.%)'),
            )
            fig.update_coloraxes(
                colorbar_tickformat=".2e"
            )
            # Display in Streamlit
            st.plotly_chart(fig, use_container_width=True)

        # Set 2 columns for the heatmaps of lattice distortion and atomic distortion
        col7, col8 = st.columns(2, border=True, gap='medium')

        # Set the 1st column for lattice distortion
        with col7:
            st.markdown('**Lattice Distortion (%):**')
            # Create a heatmap for lattice distortion
            heatmap_lattice = data_dis.pivot_table(index='Dopant Concentration A (at.%)', columns='Dopant Concentration B (at.%)', values='Lattice Distortion (%)')
            # Create the Plotly heatmap
            fig = px.imshow(
                heatmap_lattice,
                color_continuous_scale="Viridis",
                labels=dict(x=f'{dopant_B} (at.%)', y=f'{dopant_A} (at.%)'),
            )
            fig.update_coloraxes(
                colorbar_tickformat=".2f"
            )
            # Display in Streamlit
            st.plotly_chart(fig, use_container_width=True)

        # Set the 2nd column for atomic distortion
        with col8:
            st.markdown('**Atomic Distortion (Å):**')
            # Create a heatmap for atomic distortion
            heatmap_atomic = data_dis.pivot_table(index='Dopant Concentration A (at.%)', columns='Dopant Concentration B (at.%)', values='Atomic Distortion (Å)')
            # Create the Plotly heatmap
            fig = px.imshow(
                heatmap_atomic,
                color_continuous_scale="Viridis",
                labels=dict(x=f'{dopant_B} (at.%)', y=f'{dopant_A} (at.%)'),
            )
            fig.update_coloraxes(
                colorbar_tickformat=".2f"
            )
            # Display in Streamlit
            st.plotly_chart(fig, use_container_width=True)

    # Add a footer
    st.divider()
    st.markdown('**Copyright © 2025 Guangchen Liu. All rights reserved.**')


if __name__ == '__main__':
    main()