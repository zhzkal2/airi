/**
 * Agora Real-Time Transcription (RTT) provider
 *
 * Uses Agora RTC channel + STT REST API for real-time speech-to-text.
 * Requires: App ID, Customer ID, Customer Secret from Agora Console.
 */

import type { TranscriptionProviderWithExtraOptions } from '@xsai-ext/providers/utils'

import type { AgoraStreamTranscriptionExtraOptions } from './stream-transcription'

export type { AgoraStreamTranscriptionExtraOptions } from './stream-transcription'
export { streamAgoraTranscription } from './stream-transcription'

export function createAgoraRTTProvider(
  appId: string,
  customerId: string,
  customerSecret: string,
  options?: { language?: string },
): TranscriptionProviderWithExtraOptions<string, AgoraStreamTranscriptionExtraOptions> {
  return {
    transcription: (model: string, extraOptions?: AgoraStreamTranscriptionExtraOptions) => {
      return {
        baseURL: 'https://api.agora.io',
        model: model || 'agora-rtt-v1',
        fetch: async () => {
          // Agora STT does not use HTTP fetch for transcription —
          // it uses RTC data messages. This is a placeholder to satisfy the interface.
          // The actual streaming is handled by streamAgoraTranscription.
          throw new Error('Agora RTT does not support direct fetch-based transcription. Use the streaming API.')
        },
        // Pass credentials and options through for streamAgoraTranscription to use
        credentials: {
          appId,
          customerId,
          customerSecret,
        },
        language: extraOptions?.language || options?.language || 'en-US',
        abortSignal: extraOptions?.abortSignal,
        localUid: extraOptions?.localUid,
        subBotUid: extraOptions?.subBotUid,
        pubBotUid: extraOptions?.pubBotUid,
      } as any
    },
  }
}
