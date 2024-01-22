import torch


def default_sim(shot_embeds, text_embeds):
    # Assumes (n_frames, embed_dim)
    shot_embeds = torch.mean(shot_embeds, dim = 0).unsqueeze(0)
    shot_embeds = torch.nn.functional.normalize(shot_embeds, dim=-1)
        
    sim = torch.mm(text_embeds.cpu(), shot_embeds.T).float()[0][0].item()
    return sim


def sim1(shot_embeds, text_embeds):

    n_frames, n_queries, embed_dim = shot_embeds.shape
    n_texts, embed_dim = text_embeds.shape

    shot_embeds = torch.nn.functional.normalize(shot_embeds, dim=-1)

    # n_frames, n_queries, 1
    frame_sims = torch.matmul(shot_embeds, text_embeds.T)

    # n_frames, 1
    max_sim_idx = torch.argmax(frame_sims, dim=1)

    # n_frames, 1, embed_dim
    max_sim_idx = max_sim_idx.expand((n_frames, embed_dim)).unsqueeze(1)

    # n_frames, embed_dim
    sim_shot_embeds = torch.gather(shot_embeds, dim=1, index=max_sim_idx).squeeze(1)

    # embed_dim
    shot_embed = sim_shot_embeds.mean(dim=0)

    # (1, embed_dim) x (embed_dim, n_texts) = (1, n_texts)
    final_sim = torch.matmul(shot_embed.squeeze(0), text_embeds.T)
    return final_sim[0].detach().float().item()


def sim2(shot_embeds, text_embeds):

    shot_embeds = torch.nn.functional.normalize(shot_embeds, dim=-1)
    # n_frames, n_queries, embed_dim x embed_dim, 1
    # => n_frames, n_queries, 1
    frame_sims = torch.matmul(shot_embeds, text_embeds.T)

    return frame_sims.squeeze(2).max(dim=1).values.mean().detach().float().item()


def sim3(shot_embeds, text_embeds):
    # n_frames, n_queries, embed_dim  -> n_queries, embed_dim
    shot_embeds = torch.nn.functional.normalize(shot_embeds, dim=-1)
    shot_embeds_avg = shot_embeds.mean(dim=0)

    # n_queries, embed_dim x embed_dim, 1 -> n_queries, 1
    query_sims = torch.matmul(shot_embeds_avg, text_embeds.T)

    return query_sims.max().detach().float().item()


def sim4(shot_embeds, text_embeds):
    # n_frames, n_queries, embed_dim  x embed_dim, 1
    # => n_frames, n_queries, 1
    shot_embeds = torch.nn.functional.normalize(shot_embeds, dim=-1)
    frame_sims = torch.matmul(shot_embeds, text_embeds.T)
    return frame_sims.max().detach().float().item()

SIMILARITIES = {
    'default': default_sim,
    'sim1': sim1,
    'sim2': sim2,
    'sim3': sim3,
    'sim4': sim4
}