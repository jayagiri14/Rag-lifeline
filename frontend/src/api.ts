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

export interface HistoryInsightSource {
  summary: string;
  date?: string;
  is_chronic: boolean;
  type?: string;
  score: number;
  raw_text?: string;
}

export interface HistoryInsightResponse {
  insight: string;
  history_used: HistoryInsightSource[];
  model: string;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
  disclaimer: string;
}

export interface PrescriptionUploadResponse {
  status: string;
  patient_id: string;
  stored: number;
  engine: string;
  structured: Record<string, unknown>;
}

export interface AudioUploadResponse {
  status: string;
  patient_id: string;
  transcript: string;
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
  },

  async uploadPrescription(
    patientId: string,
    file: File
  ): Promise<PrescriptionUploadResponse> {
    const formData = new FormData();
    formData.append('patient_id', patientId);
    formData.append('file', file);

    const response = await axios.post(
      `${API_BASE_URL}/history/prescription`,
      formData,
      { headers: { 'Content-Type': 'multipart/form-data' } }
    );
    return response.data;
  },

  async uploadAudio(
    patientId: string,
    audio: Blob
  ): Promise<AudioUploadResponse> {
    const formData = new FormData();
    formData.append('patient_id', patientId);
    formData.append('file', audio, 'audio.webm');

    const response = await axios.post(
      `${API_BASE_URL}/history/audio`,
      formData,
      { headers: { 'Content-Type': 'multipart/form-data' } }
    );
    return response.data;
  },

  async historyInsight(
    patientId: string,
    symptoms: string,
    topK = 6
  ): Promise<HistoryInsightResponse> {
    const response = await axios.post(`${API_BASE_URL}/history/insight`, {
      patient_id: patientId,
      symptoms,
      top_k: topK,
    });
    return response.data;
  }
};
