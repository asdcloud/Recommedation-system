import torch
import tqdm

def evaluate(model, evaluate_data, epoch_id, writer, metron, use_cuda, batchify_eval, batch_size):
    """評估模型性能"""
    model.eval()
    with torch.no_grad():
        test_users, test_items = evaluate_data[0], evaluate_data[1]
        negative_users, negative_items = evaluate_data[2], evaluate_data[3]
        if use_cuda:
            test_users, test_items = test_users.cuda(), test_items.cuda()
            negative_users, negative_items = negative_users.cuda(), negative_items.cuda()

        if not batchify_eval:    
            test_scores = model(test_users, test_items)
            negative_scores = model(negative_users, negative_items)
        else:
            test_scores = []
            negative_scores = []
            for start_idx in range(0, len(test_users), batch_size):
                end_idx = min(start_idx + batch_size, len(test_users))
                batch_test_users = test_users[start_idx:end_idx]
                batch_test_items = test_items[start_idx:end_idx]
                test_scores.append(model(batch_test_users, batch_test_items))
            for start_idx in tqdm(range(0, len(negative_users), batch_size)):
                end_idx = min(start_idx + batch_size, len(negative_users))
                batch_negative_users = negative_users[start_idx:end_idx]
                batch_negative_items = negative_items[start_idx:end_idx]
                negative_scores.append(model(batch_negative_users, batch_negative_items))
            test_scores = torch.cat(test_scores, dim=0)
            negative_scores = torch.cat(negative_scores, dim=0)

            if use_cuda:
                test_users, test_items, test_scores = test_users.cpu(), test_items.cpu(), test_scores.cpu()
                negative_users, negative_items, negative_scores = negative_users.cpu(), negative_items.cpu(), negative_scores.cpu()
        
        metron.subjects = [test_users.data.view(-1).tolist(),
                           test_items.data.view(-1).tolist(),
                           test_scores.data.view(-1).tolist(),
                           negative_users.data.view(-1).tolist(),
                           negative_items.data.view(-1).tolist(),
                           negative_scores.data.view(-1).tolist()]
        
        hit_ratio = metron.cal_hit_ratio()
        ndcg = metron.cal_ndcg()
        writer.add_scalar('performance/HR', hit_ratio, epoch_id)
        writer.add_scalar('performance/NDCG', ndcg, epoch_id)
        print('[Evaluating Epoch {}] HR = {:.4f}, NDCG = {:.4f}'.format(epoch_id, hit_ratio, ndcg))
        
    return hit_ratio, ndcg

