/**
 * Agora Real-Time Transcription (RTT) REST API client
 *
 * API v7.x — https://api.agora.io/api/speech-to-text/v1/projects/{appId}
 * Auth: HTTP Basic with Customer ID / Customer Secret
 */

const AGORA_API_BASE = 'https://api.agora.io/api/speech-to-text/v1/projects'

export interface AgoraSTTCredentials {
  appId: string
  customerId: string
  customerSecret: string
}

export interface AgoraJoinRequest {
  languages: string[]
  maxIdleTime?: number
  rtcConfig: {
    channelName: string
    subBotUid: string
    pubBotUid: string
    subBotToken?: string
    pubBotToken?: string
    subscribeAudioUids?: string[]
    enableJsonProtocol?: boolean
  }
  translateConfig?: {
    languages: Array<{ source: string, target: string[] }>
  }
}

export interface AgoraJoinResponse {
  agent_id: string
  create_ts: number
  status: AgoraAgentStatus
}

export type AgoraAgentStatus = 'IDLE' | 'STARTING' | 'RUNNING' | 'STOPPING' | 'STOPPED' | 'RECOVERING' | 'FAILED'

export interface AgoraQueryResponse {
  agent_id: string
  create_ts: number
  status: AgoraAgentStatus
}

function basicAuthHeader(customerId: string, customerSecret: string): string {
  const encoded = btoa(`${customerId}:${customerSecret}`)
  return `Basic ${encoded}`
}

export async function agoraSTTJoin(
  credentials: AgoraSTTCredentials,
  request: AgoraJoinRequest,
): Promise<AgoraJoinResponse> {
  const url = `${AGORA_API_BASE}/${credentials.appId}/join`
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Authorization': basicAuthHeader(credentials.customerId, credentials.customerSecret),
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  })

  if (!response.ok) {
    const text = await response.text().catch(() => '')
    throw new Error(`Agora STT join failed (${response.status}): ${text}`)
  }

  return response.json()
}

export async function agoraSTTQuery(
  credentials: AgoraSTTCredentials,
  agentId: string,
): Promise<AgoraQueryResponse> {
  const url = `${AGORA_API_BASE}/${credentials.appId}/agents/${agentId}`
  const response = await fetch(url, {
    method: 'GET',
    headers: {
      Authorization: basicAuthHeader(credentials.customerId, credentials.customerSecret),
    },
  })

  if (!response.ok) {
    const text = await response.text().catch(() => '')
    throw new Error(`Agora STT query failed (${response.status}): ${text}`)
  }

  return response.json()
}

export async function agoraSTTLeave(
  credentials: AgoraSTTCredentials,
  agentId: string,
): Promise<void> {
  const url = `${AGORA_API_BASE}/${credentials.appId}/agents/${agentId}/leave`
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      Authorization: basicAuthHeader(credentials.customerId, credentials.customerSecret),
    },
  })

  if (!response.ok) {
    const text = await response.text().catch(() => '')
    throw new Error(`Agora STT leave failed (${response.status}): ${text}`)
  }
}
