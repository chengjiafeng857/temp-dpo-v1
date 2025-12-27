import torch
import torch.nn.functional as F

# calculate the log probability
def compute_log_prob(logits, labels):
    """ Compute the log probability for each token:

    logits: the output without softmax computation, shape: (batch_size, seq_length, vocab_size)

    labels: labels` is the "supervised version" of `input_ids to help the model 
    The purpose of labels is to help the model correctly align the responses.
    input_ids: [prompt, response], labels only care about response, so the prompt part is mask [mask, response]
    labels.shape: (batch_size, seq_length)
    """
    logits = logits[:, :-1, :]
    labels = labels[:, 1:].clone()

    loss_mask = (labels != -100)
    labels[labels == -100] = 0

    per_token_log_prob = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

   
    return (per_token_log_prob * loss_mask).sum(-1)
    
# calculate the dpo loss
# we will do the loss.mean() in the training step
def dpo_loss(policy_chosen_log_prob, policy_rejected_log_prob, ref_chosen_log_prob, ref_rejected_log_prob, beta):
    
    chosen_log_prob = policy_chosen_log_prob - ref_chosen_log_prob
    rejected_log_prob = policy_rejected_log_prob - ref_rejected_log_prob

    # record the M that we need to calculate
    # we can detach it in training step
    policy_diff = policy_chosen_log_prob - policy_rejected_log_prob
    ref_diff =  ref_chosen_log_prob - ref_rejected_log_prob
    model_margin = (policy_diff - ref_diff).detach()
    
    # compute the loss
    loss = - F.logsigmoid(beta * (chosen_log_prob - rejected_log_prob))

    chosen_rewards = (beta * chosen_log_prob).detach()
    rejected_rewards = (beta * rejected_log_prob).detach()

    return loss, chosen_rewards, rejected_rewards, model_margin

