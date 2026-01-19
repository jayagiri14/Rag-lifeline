import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export interface Source {
  content: string;
  condition: string;
  relevance_score: number;
}

export interface QueryResponse {
  response: string;
  sources: Source[];
  model: string;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
}

export interface HealthResponse {
  status: string;
  documents_loaded: number;
}

export const api = {
  async query(query: string, topK: number = 3): Promise<QueryResponse> {
    const response = await axios.post(`${API_BASE_URL}/query`, {
      query,
      top_k: topK
    });
    return response.data;
  },

  async healthCheck(): Promise<HealthResponse> {
    const response = await axios.get(`${API_BASE_URL}/health`);
    return response.data;
  },

  async reloadData(): Promise<{ status: string; documents_added: number }> {
    const response = await axios.post(`${API_BASE_URL}/reload-data`);
    return response.data;
  }
};
