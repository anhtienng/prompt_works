import torch

def generate(
    model,
    img,
    text,
    args=None
):
    model.eval()

    with torch.no_grad():
        if args.encoder_type == 'ctranspath':
            img = model.encoder(img)
        else:
            img = model.encoder(img)[1]
        img = model.projector(img)                  # bs, project_dim
        img = img.reshape(img.shape[0], -1, 512)    # bs, project_dim//512, 512
        token = model.tokenizer(text, return_tensors="pt", padding=True)   # bs, seq_len
        input_ids=token['input_ids'][:,:-1].to(args.device)    # skip the eos token

        for _ in range(args.generate_length+1):
            attention_mask=torch.where(input_ids<49407,1,0).to(args.device)    # skip the eos token
            output = model.forward_decoder(proj_encoder_feature=img,
                                   input_ids=input_ids,
                                   attention_mask=attention_mask)
            logits = model.decoder_head(output.last_hidden_state[:,-1,:])    # forward the last token embedding though a head, bs x 49408
            
            # Get a token with highest prob, and decode to get a corresponding next word
            next_token = torch.argmax(logits, -1).unsqueeze(1)               # bs x 1
                       # bs x 1

            # Append a next word to current text
            input_ids = torch.cat((input_ids,next_token),dim=1)
    result = model.tokenizer.batch_decode(input_ids)
    for i in range(len(result)):
        result[i] = result[i].split('.')[0].replace('<|startoftext|>', '').replace('<|endoftext|>', '') + '.'
    return list(result)