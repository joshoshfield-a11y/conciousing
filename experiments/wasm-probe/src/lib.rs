// profiler/frontend/wasm-probe/src/lib.rs
use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use web_sys::{window, WebSocket, MessageEvent};
use js_sys::Uint8Array;

#[wasm_bindgen]
pub struct ConciousingProbe {
    ws: WebSocket,
    agent_id: String,
    step: u64,
}

#[wasm_bindgen]
impl ConciousingProbe {
    #[wasm_bindgen(constructor)]
    pub fn new(agent_id: String, ws_url: String) -> Result<ConciousingProbe, JsValue> {
        let ws = WebSocket::new(&ws_url)?;
        ws.set_binary_type(web_sys::BinaryType::Arraybuffer);
        Ok(ConciousingProbe { ws, agent_id, step: 0 })
    }

    #[wasm_bindgen]
    pub fn step(&mut self, belief_entropy: f64, policy_entropy: f64,
                goal_depth: u32, sim_horizon: u32,
                integration: f64, surprise: f64) {
        let ct = ConciousingState {
            t: self.step,
            wall_time: js_sys::Date::now(),
            agent_id: self.agent_id.clone(),
            belief_entropy,
            policy_entropy,
            goal_stack_depth: goal_depth,
            sim_horizon,
            integration_proxy: integration,
            surprise,
        };
        self.step += 1;
        let bytes = serde_json::to_vec(&ct).unwrap();
        let _ = self.ws.send_with_u8_array(&Uint8Array::from(bytes.as_slice()));
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ConciousingState {
    t: u64,
    wall_time: f64,
    agent_id: String,
    belief_entropy: f64,
    policy_entropy: f64,
    goal_stack_depth: u32,
    sim_horizon: u32,
    integration_proxy: f64,
    surprise: f64,
}