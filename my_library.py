def test_load():
  return 'loaded'

def compute_probs(neg,pos):
  p0 = neg/(neg+pos)
  p1 = pos/(neg+pos)
  return [p0,p1]

def cond_prob(table, evidence, evidence_val, target, target_val):
  t_subset = up_table_subset(table, target, 'equals', target_val)
  e_list = up_get_column(t_subset, evidence)
  p_b_a = sum([1 if v==evidence_val else 0 for v in e_list])/len(e_list)
  return p_b_a + .01  #completely incorrect Laplace smoothing factor

def cond_probs_product(table, evidence_vals, target_col, target_val):
  evidence_cols = up_list_column_names(table)[:-1]
  evidence_complete = up_zip_lists(evidence_cols, evidence_vals)
  cond_prob_list = [cond_prob(table,column,val,target_col,target_val) for column,val in evidence_complete]
  partial_numerator = up_product(cond_prob_list)
  return partial_numerator

def prior_prob(table, col, val):
  t_list = up_get_column(table, col)
  prob = sum([1 if v==val else 0 for v in t_list])/len(t_list)
  return prob

def naive_bayes(table, evidence_row, target):
  #compute P(Flu=0|...) by collecting cond_probs in a list, take the produce of the list, finally multiply by P(Flu=0)
  p_zero = cond_probs_product(table, evidence_row, target, 0) * prior_prob(table, target, 0)

  #do same for P(Flu=1|...)
  p_one = cond_probs_product(table, evidence_row, target, 1) * prior_prob(table, target, 1)

  #Use compute_probs to get 2 probabilities
  neg, pos = compute_probs(p_zero,p_one)
  
  #return your 2 results in a list
  return [neg, pos]

