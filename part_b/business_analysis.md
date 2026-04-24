Part B: Business Case Analysis

B1. Problem Formulation 
(a) ML Problem Definition 
Target variable:
- Items Sold — the number of units sold in a store during a given month under a specific promotion.
Candidate input features:
- Store attributes: store size, location type, footfall, competition density, demographics
- Promotion attributes: promotion type, discount depth, category targeted
- Temporal features: month, season, festival periods, weekends
- Historical performance: past sales volume, past promotion effectiveness
- External factors: local events, weather, economic indicators (if available)
Type of ML problem:
- This is a supervised regression problem because the target variable (items sold) is continuous.
- The goal is to predict expected sales volume for each store–promotion–month combination so the retailer can choose the promotion that maximises items sold.
Justification:
Regression is appropriate because the business wants quantitative forecasts, not just classification of “good/bad” promotions. The model must estimate how many items will be sold under each promotion scenario.

(b) Why “Items Sold” Is a Better Target 
Revenue is influenced by price, which varies across:
- product categories
- discount levels
- seasonal pricing
- store assortment differences
Two stores may generate the same revenue but sell very different numbers of items.
For promotion optimisation, volume is the true indicator of promotional effectiveness because promotions aim to increase unit movement, not necessarily revenue.
This illustrates a broader ML principle:
Choose a target variable that directly reflects the business objective and is not confounded by unrelated factors.


(c) Why Not One Global Model? 
Stores in different locations respond differently to promotions due to:
- demographic differences
- income levels
- competition intensity
- product preferences
- store size and assortment
Better strategy:
- Use a hierarchical modelling approach, such as:
- A global model with store‑level features (store embeddings, location type, competition density), or
- Clustered models (e.g., separate models for urban, semi‑urban, rural), or
- Multi‑task learning, where each store is a task sharing information with others.
Justification:
This approach captures shared patterns across stores while still allowing store‑specific behaviour, improving accuracy and interpretability.

B2. Data and EDA Strategy 
(a) Joining the Four Tables
Tables:
- Transactions (store_id, date, items sold, promotion_id)
- Store attributes (store_id, size, location, competition density)
- Promotion details (promotion_id, promotion_type, discount depth)
- Calendar (date, weekend flag, festival flag)
Join strategy:
- Join transactions ↔ store attributes on store_id
- Join transactions ↔ promotion details on promotion_id
- Join transactions ↔ calendar on transaction_date
Final grain:
One row = one store × one day × one promotion event

Aggregations before modelling:
- Aggregate to monthly store‑level:
- Total items sold
- Count of promotion days
- Average competition density
- Most frequent promotion type
- Number of festival days
- Weekend ratio
- Create lag features:
- Previous month’s sales
- Sales under each promotion in the past 3 months
- Rolling averages
This produces a clean, modelling‑ready dataset.

(b) EDA Plan 
At least four analyses:
1. Promotion effectiveness by store type
- Chart: bar plot of average items sold by promotion × store size
- Purpose: identify which promotions work best in small vs medium vs large stores
- Influence: informs feature interactions and potential segmentation
2. Seasonality analysis
- Chart: monthly sales trend over 3 years
- Purpose: detect seasonal peaks (festivals, holidays)
- Influence: motivates adding month, festival, and lag features
3. Competition density vs sales
- Chart: scatter plot or heatmap
- Purpose: understand whether high competition reduces promotion impact
- Influence: include competition density as a key feature
4. Promotion mix imbalance
- Chart: distribution of promotion types
- Purpose: detect under‑represented promotions
- Influence: may require resampling or regularisation to avoid bias
5. Correlation matrix
- Purpose: identify multicollinearity
- Influence: informs feature selection and model choice

(c) Impact of Promotion Imbalance 
If 80% of transactions occur without promotions, the model may:
- Learn to always predict “no promotion” scenarios
- Underestimate the effect of promotions
- Fail to distinguish between promotion types
Mitigation strategies:
- Stratified sampling to ensure balanced training
- Upsampling minority promotion types
- Promotion‑specific features (e.g., one‑hot encoding)
- Regularisation to prevent dominance of the majority class
- Model evaluation by promotion segment, not just overall RMSE

B3. Model Evaluation and Deployment 
(a) Train–Test Split & Metrics
Data: 3 years × 50 stores = 1800 monthly records.
Train–test split:
- Use time‑based split:
- Train: Years 1–2
- Test: Year 3
- Random split is inappropriate because it leaks future information into the past, artificially inflating performance.
Evaluation metrics:
- RMSE — penalises large errors; important for forecasting spikes
- MAE — interpretable average error; stable against outliers
- MAPE — percentage error; useful for comparing stores of different sizes
- R² — proportion of variance explained; good for overall model fit
Interpretation:
- RMSE tells us how badly we miss during high‑volume months
- MAE tells us typical error per store per month
- MAPE helps compare performance across stores with different sales levels

(b) Explaining Different Recommendations 
To explain why Store 12 gets different promotions in December vs March:
- Extract feature importance from the model (e.g., Random Forest or SHAP values).
- Compare the top contributing features for December vs March predictions.
- Communicate insights in business language:
Example explanation:
- “In December, festival season and high footfall increase the value of Loyalty Points, which historically drive repeat purchases.”
- “In March, competition density is higher and customers respond better to Flat Discounts, which the model identifies as the strongest driver of volume.”
This approach builds trust by showing that recommendations are data‑driven and context‑specific, not arbitrary.

(c) Deployment Strategy 
1. Saving the model
- Save the trained pipeline using joblib or pickle
- Store versioned models in a central repository (e.g., Azure Blob, S3)
2. Monthly scoring workflow
At the start of each month:
- Pull latest store attributes, promotions, and calendar data
- Apply the same preprocessing pipeline used during training
- Generate predictions for each store × promotion combination
- Select the promotion with the highest predicted items sold
3. Monitoring
Monitor:
- Prediction drift — distribution of predicted values vs historical
- Feature drift — changes in store attributes or customer behaviour
- Actual vs predicted error — rising RMSE indicates degradation
- Promotion effectiveness — compare model‑recommended promotions vs actual outcomes
4. Retraining triggers
Retrain when:
- Drift exceeds thresholds
- New promotions are introduced
- Store demographics change
- Model performance drops below acceptable levels
This ensures the model remains accurate and aligned with evolving retail dynamics.

