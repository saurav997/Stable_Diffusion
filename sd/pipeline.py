import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

Width,Hieght = 512,512
Latent_Width = Width//8
Latent_Hieght = Hieght//8

def generate(prompt:str, 
             uncond_prompt:str, # the negative prompt or an empty string
             input_image=None, 
             strength =0.8, 
             do_cfg=True,
             cfg_scale=7.5,
             sampler_name="ddpm",
             n_inference_Steps = 50, 
             models = {},
             seed =None,
             device =None,
             idle_device =None, 
             tokenizer =None
             ):

    with torch.no_grad():
        if not(0<strength<=1):
            raise ValueError("strength must be between 0 and 1!")
        
        if idle_device:
            to_idle: lambda x: x.to(idle_device)
        else:
            to_idle: lambda x: x
        generator = torch.Generator(device = device)
        if seed is None:
            generate.seed()
        else:
            generator.manual_seed(seed)
        
        clip = models["clip"]
        clip = clip.to(device)
        # if we are doing Classifier-free guidance training
        if do_cfg:
            # Convert the prompt input into tokens using the tokenizer
            cond_tokens = tokenizer.batch_encode_plus([prompt],padding="max_length",max_length=77).input_ids
            # (Batch_size,Seq_Len)
            cond_tokens = torch.tensor(cond_tokens,dtype = torch.long,device=device)
            # (Bath_size, Seq_Len,embed_dim)
            cond_context = clip(cond_tokens)

            
            # create an empty prompt that acts as an input when no prompt is given
            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt],padding="max_length",max_length=77).input_ids
            uncond_tokens = torch.tensor(uncond_tokens,dtype = torch.long,device=device)
            # (Bath_size, Seq_Len,embed_dim)
            uncond_context = clip(uncond_tokens)

            #append them into a single prompt  
            context = torch.cat([cond_context, uncond_context])
        # if Classifier-free guidance training is disabled
        else:
            tokens = tokenizer.batch_encode_plus([prompt],padding = "max_length",max_length = 77).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long,device = device)
            context = clip(tokens)
        to_idle(clip)

        if sampler_name=="ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")  
        
        latents_shape = (1,4,Latent_Hieght,Latent_Width)
        if input_image:
            encoder = models["encoder"]
            encoder.to(device)
            input_image_tensor = input_image.resize((Latent_Width,Latent_Hieght))
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            input_image_tensor = rescale(input_image_tensor,(0,255),(-1,1))
            # (Hieght,Width,Channel) -> (Batch_size =1 ,Hieght,Width,Channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (Batch_size =1 ,Hieght,Width,Channel) -> (Batch_size =1 ,channel,Hieght,Width)
            input_image_tensor = input_image_tensor.permute(0,3,1,2)
            
            encoder_noise = torch.randn(latents_shape, generator = generator,device = device)
            # Getting the compressed latents by passing input image and noise via VAE encoder
            latents = encoder(input_image_tensor, encoder_noise)
            sampler.set_strength(strength = strength)
            latents = sampler.add_noise(latents, sampler.timestep[0])
            to_idle(encoder)

        else:
            # If we are doing text-to-image, we start with random noise sampled from N(0,I)
            latents = torch.randn(latents_shape, generator = generator,device = device)
            diffusion = models["diffusion"]
            diffusion.to(device)
            timesteps = tqdm(sampler.timesteps)
            for i,timestep in enumerate(timesteps):
                # converting the particular timestep to an embedding of dimension (1,320)
                time_embedding = get_time_embedding(timestep).to(device)
                # latents are of dimensions (Batch_size,4,Latents_Hieght,Latents_Width)
                model_input = latents 
                if do_cfg:
                    # (Batch_size,4,Latents_Hieght,Latents_Width) -> (2*Batch_size,4,Latents_Hieght,Latents_Width)
                    model_input = model_input.repeat(2,1,1,1)
                # model_output is the prediction of the diffusion model
                model_output = diffusion(model_input,context,time_embedding)
                if do_cfg:
                    output_cond, output_uncond = model_output.chunk(2)
                    model_output = cfg_scale*(output_cond - output_uncond) + output_uncond

                # Remove noise predicted by UNET
                latents = sampler.step(timestep, latents, model_output)
            
            to_idle(diffusion)
            decoder = models["decoder"]
            decoder.to(device)
            images = decoder(latents)
            to_idle(decoder)
            images = rescale(images,(-1,1),(0,255),clamp =True)
            # (Batch_Size,channels,Hieght,Width) -> (Batch_Size,Hieght,Width,channels)
            images = images.permute(0,2,3,1)
            images = images.to("cpu",torch.uint8).numpy()
            return images[0]

def rescale(x, old_range,new_range,clamp = False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x += (new_max - new_min)/(old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min,new_max)
    return x

def get_time_embedding(timestep):
    # (160,)
    freqs = torch.pow(10000, -torch.arange(start =0,end = 160,dtype = torch.float32)/160)
    # (1,160)
    x = torch.tensor([timestep],dtype = torch.float32)[:,None]*freqs[None]
    return torch.cat([torch.cos(x),torch.sin(x)],dim =-1)
















            



