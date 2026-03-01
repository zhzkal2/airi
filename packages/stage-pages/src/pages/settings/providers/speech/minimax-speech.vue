<script setup lang="ts">
import {
  SpeechPlayground,
  SpeechProviderSettings,
} from '@proj-airi/stage-ui/components'
import { useSpeechStore } from '@proj-airi/stage-ui/stores/modules/speech'
import { useProvidersStore } from '@proj-airi/stage-ui/stores/providers'
import { generateSpeechMiniMax } from '@proj-airi/stage-ui/utils/minimax-tts'
import { FieldInput } from '@proj-airi/ui'
import { storeToRefs } from 'pinia'
import { computed, watch } from 'vue'

const providerId = 'minimax-speech'

const speechStore = useSpeechStore()
const providersStore = useProvidersStore()
const { providers } = storeToRefs(providersStore)
const { activeSpeechProvider } = storeToRefs(speechStore)

const isConfigured = computed(() => !!providers.value[providerId]?.apiKey && !!providers.value[providerId]?.groupId)
const apiKeyConfigured = computed(() => !!providers.value[providerId]?.apiKey)
const isActive = computed(() => activeSpeechProvider.value === providerId)

const groupId = computed({
  get: () => providers.value[providerId]?.groupId as string | undefined || '',
  set: (value) => {
    if (!providers.value[providerId])
      providers.value[providerId] = {}
    providers.value[providerId].groupId = value
  },
})

const availableVoices = computed(() => {
  return speechStore.availableVoices[providerId] || []
})

function setAsActiveProvider() {
  activeSpeechProvider.value = providerId
}

async function handleGenerateSpeech(input: string, voiceId: string, _useSSML: boolean): Promise<ArrayBuffer> {
  const providerConfig = providersStore.getProviderConfig(providerId)
  return await generateSpeechMiniMax({
    text: input,
    voiceId,
    apiKey: providerConfig?.apiKey as string | undefined,
    groupId: providerConfig?.groupId as string | undefined,
  })
}

watch(providers, async () => {
  await speechStore.loadVoicesForProvider(providerId)
}, { immediate: true })
</script>

<template>
  <SpeechProviderSettings
    :provider-id="providerId"
  >
    <template #basic-settings>
      <FieldInput
        v-model="groupId"
        label="Group ID"
        description="MiniMax Group ID (required for TTS API)"
        required
      />
      <div flex="~ col gap-2" mt-2>
        <button
          :disabled="!isConfigured"
          border="~ rounded-lg"
          px-4 py-2 text-sm font-medium transition-colors
          :class="isActive
            ? 'border-green-500 bg-green-500/10 text-green-600 dark:text-green-400 cursor-default'
            : isConfigured
              ? 'border-primary-500 bg-primary-500/10 text-primary-600 dark:text-primary-400 hover:bg-primary-500/20 cursor-pointer'
              : 'border-neutral-300 bg-neutral-100 text-neutral-400 dark:border-neutral-700 dark:bg-neutral-800 cursor-not-allowed'"
          @click="!isActive && setAsActiveProvider()"
        >
          <span v-if="isActive">✓ Active Speech Provider</span>
          <span v-else-if="isConfigured">Set as Active Speech Provider</span>
          <span v-else>Enter API Key and Group ID to activate</span>
        </button>
      </div>
    </template>

    <template #playground>
      <SpeechPlayground
        :available-voices="availableVoices"
        :generate-speech="handleGenerateSpeech"
        :api-key-configured="apiKeyConfigured"
        default-text="Hello! This is a test of the MiniMax voice synthesis."
      />
    </template>
  </SpeechProviderSettings>
</template>

<route lang="yaml">
  meta:
    layout: settings
    stageTransition:
      name: slide
  </route>
