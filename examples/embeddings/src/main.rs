//! This is a translation of embedding.cpp in llama.cpp using llama-cpp-2.
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use clap::Parser;
use hf_hub::api::sync::ApiBuilder;

use llama_cpp_4::context::params::LlamaContextParams;
use llama_cpp_4::context::LlamaContext;
use llama_cpp_4::ggml_time_us;
use llama_cpp_4::llama_backend::LlamaBackend;
use llama_cpp_4::llama_batch::LlamaBatch;
use llama_cpp_4::model::params::LlamaModelParams;
use llama_cpp_4::model::LlamaModel;
use llama_cpp_4::model::{AddBos, Special};

#[derive(clap::Parser, Debug, Clone)]
struct Args {
    /// The path to the model
    #[command(subcommand)]
    model: Model,
    /// The prompt
    #[clap(default_value = "Hello my name is\nWhat is your name?")]
    prompt: String,
    /// Whether to normalise the produced embeddings
    #[clap(short)]
    normalise: bool,
    /// Disable offloading layers to the gpu
    #[cfg(any(feature = "cuda", feature = "vulkan"))]
    #[clap(long)]
    disable_gpu: bool,
}

#[derive(clap::Subcommand, Debug, Clone)]
enum Model {
    /// Use an already downloaded model
    Local {
        /// The path to the model. e.g. `/home/marcus/.cache/huggingface/hub/models--TheBloke--Llama-2-7B-Chat-GGUF/blobs/08a5566d61d7cb6b420c3e4387a39e0078e1f2fe5f055f3a03887385304d4bfa`
        path: PathBuf,
    },
    /// Download a model from huggingface (or use a cached version)
    #[clap(name = "hf-model")]
    HuggingFace {
        /// the repo containing the model. e.g. `BAAI/bge-small-en-v1.5`
        repo: String,
        /// the model name. e.g. `BAAI-bge-small-v1.5.Q4_K_M.gguf`
        model: String,
    },
}

impl Model {
    /// Convert the model to a path - may download from huggingface
    fn get_or_load(self) -> Result<PathBuf> {
        match self {
            Model::Local { path } => Ok(path),
            Model::HuggingFace { model, repo } => ApiBuilder::new()
                .with_progress(true)
                .build()
                .with_context(|| "unable to create huggingface api")?
                .model(repo)
                .get(&model)
                .with_context(|| "unable to download model"),
        }
    }
}

fn main() -> Result<()> {
    let Args {
        model,
        prompt,
        normalise,
        #[cfg(any(feature = "cuda", feature = "vulkan"))]
        disable_gpu,
    } = Args::parse();

    // init LLM
    let mut backend = LlamaBackend::init()?;
    // don't print logs
    backend.void_logs();

    // offload all layers to the gpu
    let model_params = {
        #[cfg(any(feature = "cuda", feature = "vulkan"))]
        if !disable_gpu {
            LlamaModelParams::default().with_n_gpu_layers(1000)
        } else {
            LlamaModelParams::default()
        }
        #[cfg(not(any(feature = "cuda", feature = "vulkan")))]
        LlamaModelParams::default()
    };

    let n_batch = 2048;
    let n_ubatch = n_batch;

    let model_path = model
        .get_or_load()
        .with_context(|| "failed to get model from args")?;

    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
        .with_context(|| "unable to load model")?;

    // initialize the context
    let ctx_params = LlamaContextParams::default()
        .with_n_batch(n_batch)
        .with_n_ubatch(n_ubatch)
        .with_n_threads_batch(std::thread::available_parallelism()?.get().try_into()?)
        .with_embeddings(true);

    let mut ctx = model
        .new_context(&backend, ctx_params)
        .with_context(|| "unable to create the llama_context")?;

    // Split the prompt to display the batching functionality
    let prompt_lines = prompt.lines();

    // tokenize the prompt
    let tokens_lines_list = prompt_lines
        .map(|line| model.str_to_token(line, AddBos::Always))
        .collect::<Result<Vec<_>, _>>()
        .with_context(|| format!("failed to tokenize {prompt}"))?;

    let n_ctx = ctx.n_ctx() as usize;
    let n_ctx_train = model.n_ctx_train();
    let pooling_type = ctx.pooling_type();

    eprintln!("n_ctx = {n_ctx}, n_ctx_train = {n_ctx_train}, pooling_type = {pooling_type:?}");

    if tokens_lines_list.iter().any(|tok| n_ctx < tok.len()) {
        bail!("One of the provided prompts exceeds the size of the context window");
    }

    // print the prompt token-by-token
    eprintln!();

    for (i, token_line) in tokens_lines_list.iter().enumerate() {
        eprintln!("Prompt {i}");
        for token in token_line {
            // Attempt to convert token to string and print it; if it fails, print the token instead
            match model.token_to_str(*token, Special::Tokenize) {
                Ok(token_str) => eprintln!("{token} --> {token_str}"),
                Err(e) => {
                    eprintln!("Failed to convert token to string, error: {e}");
                    eprintln!("Token value: {token}");
                }
            }
        }
        eprintln!();
    }

    std::io::stderr().flush()?;

    // create a llama_batch with the size of the context
    // we use this object to submit token data for decoding
    let mut batch = LlamaBatch::new(n_batch as usize, 1);

    let mut max_seq_id_batch = 0;
    let mut output = Vec::with_capacity(tokens_lines_list.len());

    let t_main_start = ggml_time_us();

    for tokens in &tokens_lines_list {
        // Flush the batch if the next prompt would exceed our batch size
        // if (batch.n_tokens() as usize + tokens.len()) > n_ctx {
        if (batch.n_tokens() as usize + tokens.len()) > n_batch as usize {
            let _ = batch_decode(
                &mut ctx,
                &mut batch,
                max_seq_id_batch,
                &mut output,
                normalise,
            );
            max_seq_id_batch = 0;
        }

        batch.add_sequence(tokens, max_seq_id_batch, false)?;
        max_seq_id_batch += 1;
    }
    // Handle final batch
    batch_decode(
        &mut ctx,
        &mut batch,
        max_seq_id_batch,
        &mut output,
        normalise,
    )?;

    let t_main_end = ggml_time_us();

    for (i, embeddings) in output.iter().enumerate() {
        eprintln!("Embeddings {i}: {embeddings:?}");
        eprintln!();
    }

    let prompt_lines: Vec<&str> = prompt.lines().collect();
    if output.len() > 1 {
        println!("cosine similarity matrix:\n\n");
        prompt_lines
            .iter()
            .map(|str| {
                print!("{str:?}\t"); // cut and only print first 6 symbols
            })
            .for_each(drop);

        println!("");

        for i in 0..output.len() {
            let i_embeddings = output.get(i).unwrap();
            for j in 0..output.len() {
                let j_embeddings = output.get(j).unwrap();
                let sim = common_embd_similarity_cos(i_embeddings, j_embeddings);
                print!("{sim}\t");
            }
            let prompt = prompt_lines.get(i).unwrap();
            print!("{prompt:?}\n");
        }
    }

    let duration = Duration::from_micros((t_main_end - t_main_start) as u64);
    let total_tokens: usize = tokens_lines_list.iter().map(Vec::len).sum();
    eprintln!(
        "Created embeddings for {} tokens in {:.2} s, speed {:.2} t/s\n",
        total_tokens,
        duration.as_secs_f32(),
        total_tokens as f32 / duration.as_secs_f32()
    );

    println!("{}", ctx.timings());

    Ok(())
}

fn batch_decode(
    ctx: &mut LlamaContext,
    batch: &mut LlamaBatch,
    s_batch: i32,
    output: &mut Vec<Vec<f32>>,
    normalise: bool,
) -> Result<()> {
    let pooling_type = ctx.pooling_type();

    ctx.clear_kv_cache();
    ctx.decode(batch).with_context(|| "llama_decode() failed")?;

    for i in 0..batch.n_tokens() {
        let embedding = match pooling_type {
            llama_cpp_4::context::params::LlamaPoolingType::None => ctx.embeddings_ith(i),
            _ => ctx.embeddings_seq_ith(i),
        };

        // .with_context(|| "Failed to get embeddings")?;

        if let Ok(embedding) = embedding {
            let output_embeddings = if normalise {
                normalize(embedding)
            } else {
                embedding.to_vec()
            };

            output.push(output_embeddings);
        }
    }

    batch.clear();

    Ok(())
}

fn normalize(input: &[f32]) -> Vec<f32> {
    let magnitude = input
        .iter()
        .fold(0.0, |acc, &val| val.mul_add(val, acc))
        .sqrt();

    input.iter().map(|&val| val / magnitude).collect()
}

fn common_embd_similarity_cos(embd1: &Vec<f32>, embd2: &Vec<f32>) -> f32 {
    let mut sum = 0.0;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;

    // Iterate through the vectors
    for i in 0..embd1.len() {
        sum += embd1[i] as f64 * embd2[i] as f64;
        sum1 += embd1[i] as f64 * embd1[i] as f64;
        sum2 += embd2[i] as f64 * embd2[i] as f64;
    }

    // Handle the case where one or both vectors are zero vectors
    if sum1 == 0.0 || sum2 == 0.0 {
        if sum1 == 0.0 && sum2 == 0.0 {
            return 1.0; // Two zero vectors are considered similar
        }
        return 0.0; // One of the vectors is a zero vector
    }

    // Calculate cosine similarity
    return (sum / (f64::sqrt(sum1) * f64::sqrt(sum2))) as f32;
}
